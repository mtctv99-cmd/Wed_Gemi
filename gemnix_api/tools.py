"""Tool calling and multimodal message parsing."""
import json
import re
import uuid
import base64
import io

MAX_IMAGE_B64_SIZE = 50000  # ~37KB raw image


def _compress_b64_if_needed(b64: str) -> str:
    """Compress image if base64 is too large for text embedding."""
    if len(b64) <= MAX_IMAGE_B64_SIZE:
        return b64
    try:
        from PIL import Image
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data))
        # Resize to max 256px on longest side
        max_dim = 256
        ratio = min(max_dim / img.width, max_dim / img.height)
        if ratio < 1:
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        # Convert to JPEG with quality reduction
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=60)
        compressed = base64.b64encode(buf.getvalue()).decode()
        return compressed
    except Exception:
        # If PIL not available, truncate (model will get partial data)
        return b64[:MAX_IMAGE_B64_SIZE]


def _build_tool_choice_instruction(tool_choice, tool_defs: list) -> str:
    """Build tool_choice constraint instruction."""
    if tool_choice == "none":
        return "\nIf tools can help, please respond with text only — no tool calls needed."
    if tool_choice == "required":
        return "\nPlease call one of the available tools for this request."
    if isinstance(tool_choice, dict):
        fn_name = tool_choice.get("function", {}).get("name", "")
        if fn_name:
            return f'\nPlease call the tool "{fn_name}".'
    return ""


_BUILD_TOOL_PROMPT = """\
You have access to tools you can call. Available tools and their schemas:

{tool_spec}

To call a tool, include a code block with language "tool_call" containing JSON:
```tool_call
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
```

Call format:
```tool_call
{{"name": "<tool_name>", "arguments": {{"<arg_key>": "<arg_value>"}}}}
```

Examples:

Example 1 — call a tool:
User: "What is the weather in Tokyo?"
Assistant:
```tool_call
{{"name": "get_weather", "arguments": {{"location": "Tokyo"}}}}
```

Example 2 — call multiple independent tools:
User: "What is the weather in London and the capital of France?"
Assistant:
```tool_call
{{"name": "get_weather", "arguments": {{"location": "London"}}}}
```
```tool_call
{{"name": "search_web", "arguments": {{"query": "capital of France"}}}}
```

Example 3 — text-only response (no tool call):
User: "Hello, how are you?"
Assistant: I am doing well, thank you! How can I help you today?

Rules:
- Call tools when they would help complete the task
- When calling tools, output ONLY the tool_call block(s), no extra text
- Call multiple tools at once if they are independent
- After receiving tool results, use them to answer the user
{tool_choice}"""


def messages_to_prompt(messages: list, tools: list = None, tool_choice=None) -> tuple:
    """Convert OpenAI messages to (prompt_str, images_list)."""
    parts = []
    images = []

    if tools and tool_choice != "none":
        tool_defs = []
        for tool in tools:
            fn = tool.get("function", tool) if tool.get("type") == "function" else tool
            tool_defs.append({
                "name": fn.get("name", tool.get("name", "")),
                "description": fn.get("description", tool.get("description", "")),
                "parameters": fn.get("parameters", tool.get("parameters", {})),
            })
        if tool_defs:
            constraint = _build_tool_choice_instruction(tool_choice, tool_defs)
            parts.append(
                _BUILD_TOOL_PROMPT.format(
                    tool_spec=json.dumps(tool_defs, indent=2),
                    tool_choice=constraint,
                )
            )

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for c in content:
                if c.get("type") in ("text", "input_text"):
                    text_parts.append(c.get("text", ""))
                elif c.get("type") == "image_url":
                    url = c.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        # Base64 inline image
                        try:
                            import base64 as _b64
                            mime = url.split(";")[0].split(":")[1] if ";" in url else "image/png"
                            b64_data = url.split(",")[1] if "," in url else url
                            img_bytes = _b64.b64decode(b64_data)
                            images.append((img_bytes, mime))
                            text_parts.append("[Image attached]")
                        except Exception:
                            text_parts.append("[Image could not be decoded]")
                    else:
                        # Remote URL — download
                        try:
                            from .multimodal import fetch_image_bytes
                            img_bytes = fetch_image_bytes(url)
                            if img_bytes:
                                images.append((img_bytes, "image/png"))
                                text_parts.append("[Image attached]")
                            else:
                                text_parts.append("[Image could not be fetched]")
                        except Exception:
                            text_parts.append("[Image could not be fetched]")
                elif c.get("type") == "input_image":
                    # OpenAI Responses API format: {"type": "input_image", "image_url": "url"}
                    raw_url = c.get("image_url", "")
                    if isinstance(raw_url, dict):
                        raw_url = raw_url.get("url", "")
                    if not raw_url:
                        text_parts.append("[Image could not be decoded]")
                    elif raw_url.startswith("data:"):
                        try:
                            mime = raw_url.split(";")[0].split(":")[1] if ";" in raw_url else "image/png"
                            b64_data = raw_url.split(",")[1] if "," in raw_url else raw_url
                            img_bytes = base64.b64decode(b64_data)
                            images.append((img_bytes, mime))
                            text_parts.append("[Image attached]")
                        except Exception:
                            text_parts.append("[Image could not be decoded]")
                    else:
                        try:
                            img_bytes = fetch_image_bytes(raw_url)
                            if img_bytes:
                                images.append((img_bytes, "image/png"))
                                text_parts.append("[Image attached]")
                            else:
                                text_parts.append("[Image could not be fetched]")
                        except Exception:
                            text_parts.append("[Image could not be fetched]")
                elif c.get("type") == "image":
                    # OpenAI vision format: {"type": "image", "image_url": {...}} or {"type": "image", "data": "..."}
                    img = c.get("image_url", {}) or c.get("image", {})
                    url = ""
                    if isinstance(img, dict):
                        url = img.get("url", "")
                    if not url and isinstance(c.get("image"), str):
                        url = c["image"]
                    if url and url.startswith("data:"):
                        try:
                            import base64 as _b64
                            b64_data = url.split(",")[1] if "," in url else url
                            mime = url.split(";")[0].split(":")[1] if ";" in url else "image/png"
                            img_bytes = _b64.b64decode(b64_data)
                            images.append((img_bytes, mime))
                            text_parts.append("[Image attached]")
                        except Exception:
                            text_parts.append("[Image could not be decoded]")
                    elif url:
                        try:
                            from .multimodal import fetch_image_bytes
                            img_bytes = fetch_image_bytes(url)
                            if img_bytes:
                                images.append((img_bytes, "image/png"))
                                text_parts.append("[Image attached]")
                            else:
                                text_parts.append("[Image could not be fetched]")
                        except Exception:
                            text_parts.append("[Image could not be fetched]")
            content = "\n".join(text_parts)

        if role == "system":
            parts.append(f"[System instruction]: {content}")
        elif role == "assistant":
            if msg.get("tool_calls"):
                tc_strs = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_strs.append(
                        f'```tool_call\n{{"name": "{fn.get("name")}", '
                        f'"arguments": {fn.get("arguments", "{}")}}}\n```'
                    )
                # Show tool_call blocks only — matches expected model output format
                parts.append("[Assistant]: " + "\n".join(tc_strs))
                # Add hint for the model: what follows is a tool result
                parts.append("[System instruction]: You will receive tool results next. Use them to answer the user's question.")
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            result_text = str(content)
            if len(result_text) > 2000:
                result_text = result_text[:2000] + "\n...[truncated]"
            parts.append(f"[Tool result for {msg.get('name', '')}]: {result_text}")
        else:
            parts.append(content if content else "")

    prompt = "\n\n".join(p for p in parts if p)
    return prompt, images


def _try_extract_json_blocks(text: str) -> list:
    """Extract valid tool call JSON from multiple possible formats.

    Tries (in order):
    1. ```tool_call\\n{...}\\n``` blocks
    2. ```json\\n{"name": "...", "arguments": {...}}\\n``` blocks
    3. Bare JSON objects with "name" + "arguments"/"args" keys
    4. function_call\\n{...} (no backticks)
    """
    found = []
    patterns = [
        (r'```tool_call\s*\n(.*?)\n```', lambda d: ("name" in d)),
        (r'```json\s*\n(\{.*?"name".*?\})\n```', lambda d: ("name" in d and ("arguments" in d or "args" in d))),
        (r'(?:^|\n)function_call\s*\n(\{[^`]*?\})', lambda d: ("name" in d)),
    ]
    for pat, validator in patterns:
        for m in re.finditer(pat, text, re.DOTALL):
            try:
                data = json.loads(m.group(1).strip())
                if validator(data):
                    found.append(data)
            except (json.JSONDecodeError, ValueError):
                pass
        if found:
            break  # Stop on first pattern that matches

    # Last resort: scan for bare JSON anywhere
    if not found:
        for m in re.finditer(r'\{[^{}]*?"(?:name|function)"[^{}]*?\}', text, re.DOTALL):
            try:
                data = json.loads(m.group(0))
                if "name" in data and ("arguments" in data or "args" in data):
                    found.append(data)
            except (json.JSONDecodeError, ValueError):
                pass
            if found:
                break

    return found


def _coerce_value(value, prop_schema: dict):
    """Coerce a single value according to its JSON Schema type."""
    if not isinstance(value, str):
        return value
    target_type = prop_schema.get("type", "")

    # String -> object/array via JSON parse
    if target_type in ("object", "array"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # String -> number
    if target_type in ("number", "integer"):
        try:
            return int(value) if target_type == "integer" else float(value)
        except (ValueError, TypeError):
            pass

    # String -> boolean
    if target_type == "boolean":
        ls = value.strip().lower()
        if ls in ("true", "1", "yes"):
            return True
        if ls in ("false", "0", "no"):
            return False

    # Catch-all: string looks like JSON object/array
    stripped = value.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return value


def _coerce_arguments(args: dict, schema: dict) -> dict:
    """Validate and coerce argument values according to JSON Schema."""
    properties = schema.get("properties", {})
    if not properties:
        return args
    coerced = {}
    for key, prop_schema in properties.items():
        if key in args:
            coerced[key] = _coerce_value(args[key], prop_schema)
        elif key in schema.get("required", []):
            coerced[key] = None
    return coerced


def _find_tool_schema(name: str, tool_defs: list) -> dict | None:
    """Extract parameters JSON Schema for a named tool from tool definitions.

    Handles both OpenAI format ({type:function, function:{name,parameters}})
    and internal flat format ({name, parameters}).
    """
    for td in tool_defs:
        if not isinstance(td, dict):
            continue
        # OpenAI format
        if td.get("type") == "function":
            fn = td.get("function", {})
            if fn.get("name") == name:
                return fn.get("parameters", {})
        # Flat format
        if td.get("name") == name:
            return td.get("parameters", {})
    return None


def _tool_call_to_openai(data: dict, tool_defs: list = None) -> dict:
    """Convert parsed JSON to OpenAI tool_call format.

    When tool_defs is provided, coerce argument types against the tool schema.
    """
    args = data.get("arguments", data.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            pass
    if tool_defs and isinstance(args, dict):
        schema = _find_tool_schema(data["name"], tool_defs)
        if schema:
            args = _coerce_arguments(args, schema)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": data["name"],
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def parse_tool_calls(text: str, tool_defs: list = None) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list).

    When tool_defs is provided, coerce argument types against the tool schemas.
    """
    if not text:
        return "", []

    blocks = _try_extract_json_blocks(text)
    tool_calls = []
    for data in blocks:
        try:
            tc = _tool_call_to_openai(data, tool_defs)
            tool_calls.append(tc)
        except (KeyError, TypeError):
            pass

    # Strip matched patterns from text
    clean = text
    for pat in [r'```tool_call\s*\n.*?\n```', r'```json\s*\n\{.*?"name".*?\}\n```',
                r'(?:^|\n)function_call\s*\n\{[^`]*?\}']:
        clean = re.sub(pat, '', clean, flags=re.DOTALL).strip()

    return clean, tool_calls


# ─── Google Native API helpers ─────────────────────────────────────────────────


def build_tool_prompt(tool_defs: list) -> str:
    """Build natural tool-use prompt for Gemini Web that avoids prompt-injection detection."""
    tool_spec = json.dumps(tool_defs, indent=2, ensure_ascii=False)
    return (
        "# Tool Use\n\n"
        "You can call the following tools to help accomplish tasks. "
        "These tools connect to the user's local environment and will execute when called.\n\n"
        "Call format (use this exact format):\n"
        "```function_call\n"
        '{"name": "<tool_name>", "args": {<arguments>}}\n'
        "```\n\n"
        "When calling tools:\n"
        "- Output ONLY the function_call block(s), nothing else\n"
        "- You may call multiple tools with multiple blocks\n"
        "- After receiving a [Tool result for ...], use that data to answer the user\n\n"
        f"Available tools:\n{tool_spec}"
    )


def _google_tool_choice_instruction(req: dict) -> str:
    """Extract tool_choice constraint from Google API toolConfig."""
    tool_config = req.get("toolConfig", {})
    fc_config = tool_config.get("functionCallingConfig", {})
    mode = fc_config.get("mode", "AUTO")
    allowed = fc_config.get("allowedFunctionNames", [])

    if mode == "NONE":
        return "\n\nRespond with text only — no tool calls needed."
    if mode == "ANY":
        if allowed:
            names = ", ".join(f'"{n}"' for n in allowed)
            return f"\n\nPlease call one of these tools: {names}."
        return "\n\nPlease call at least one tool for this request."
    return ""


def google_contents_to_prompt(req: dict) -> tuple:
    """Convert Google API contents/tools/systemInstruction to (prompt_str, images_list).

    Returns (prompt, images) where images is a list of (bytes, mime_type) tuples.
    """
    parts = []
    images = []

    tool_config = req.get("toolConfig", {})
    fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")

    tools = req.get("tools")
    tool_defs = []
    if tools and fc_mode != "NONE":
        for tool_group in tools:
            for fn in tool_group.get("functionDeclarations", []):
                td = {"name": fn.get("name", ""), "description": fn.get("description", "")}
                params = fn.get("parameters") or fn.get("parametersJsonSchema")
                if params:
                    td["parameters"] = params
                tool_defs.append(td)

    sys_inst = req.get("systemInstruction")
    if sys_inst:
        sys_parts = sys_inst.get("parts", [])
        sys_text = " ".join(p.get("text", "") for p in sys_parts if p.get("text"))
        if sys_text:
            if tool_defs:
                constraint = _google_tool_choice_instruction(req)
                parts.append(sys_text + "\n\n" + build_tool_prompt(tool_defs) + constraint)
            else:
                parts.append(sys_text)
    elif tool_defs:
        constraint = _google_tool_choice_instruction(req)
        parts.append(build_tool_prompt(tool_defs) + constraint)

    for content in req.get("contents", []):
        role = content.get("role", "user")
        msg_parts = []
        for p in content.get("parts", []):
            if p.get("text"):
                msg_parts.append(p["text"])
            elif p.get("inlineData"):
                data = p["inlineData"]
                mime = data.get("mimeType", "image/png")
                images.append((base64.b64decode(data["data"]), mime))
            elif p.get("functionCall"):
                fc = p["functionCall"]
                msg_parts.append(
                    f'```function_call\n{json.dumps({"name": fc["name"], "args": fc.get("args", {})}, ensure_ascii=False)}\n```'
                )
            elif p.get("functionResponse"):
                fr = p["functionResponse"]
                msg_parts.append(
                    f'[Tool result for {fr.get("name", "")}]: {json.dumps(fr.get("response", {}), ensure_ascii=False)}'
                )
        text = "\n".join(msg_parts)
        if role == "model":
            parts.append(f"[Assistant]: {text}")
        else:
            parts.append(text)

    return "\n\n".join(p for p in parts if p), images


def parse_google_function_calls(text: str) -> tuple:
    """Extract function_call blocks from model output.

    Handles 3 formats:
    1. ```function_call\\n{...}\\n``` (standard)
    2. function_call\\n{...} (without backticks)
    3. Raw JSON with "name" + "args" keys

    Returns (clean_text, [{"name": ..., "args": ...}])
    """
    function_calls = []
    pattern1 = r'```function_call\s*\n(.*?)\n```'
    pattern2 = r'(?:^|\n)function_call\s*\n(\{[^`]*?\})'
    clean = text
    for pattern in [pattern1, pattern2]:
        for match in re.findall(pattern, clean, re.DOTALL):
            try:
                data = json.loads(match.strip())
                if "name" in data:
                    function_calls.append({
                        "name": data["name"],
                        "args": data.get("args", data.get("arguments", {})),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        clean = re.sub(pattern, '', clean, flags=re.DOTALL).strip()
    if not function_calls and clean.strip().startswith("{"):
        try:
            data = json.loads(clean.strip())
            if "name" in data and ("args" in data or "arguments" in data):
                function_calls.append({
                    "name": data["name"],
                    "args": data.get("args", data.get("arguments", {})),
                })
                clean = ""
        except (json.JSONDecodeError, KeyError):
            pass
    return clean, function_calls
