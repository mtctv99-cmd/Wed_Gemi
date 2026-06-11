"""HTTP server: OpenAI-compatible API endpoints."""
import json
import time
import uuid
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .config import CONFIG
from .models import MODELS, resolve_model
from .gemini import generate, generate_stream, log
from .tools import messages_to_prompt, parse_tool_calls, google_contents_to_prompt, parse_google_function_calls
from .multimodal import upload_image, fetch_image_bytes
from . import __version__

# Cache for Responses API multi-turn (previous_response_id)
# Expiry: 5 minutes after insert
_response_cache: dict = {}


def _cache_response(rid: str, output: list):
    _response_cache[rid] = {"output": output, "ts": time.time()}
    # Lazy-clean stale entries
    now = time.time()
    stale = [k for k, v in _response_cache.items() if now - v["ts"] > 300]
    for k in stale:
        del _response_cache[k]


def _get_cached_response(rid: str) -> list | None:
    entry = _response_cache.get(rid)
    if entry and time.time() - entry["ts"] <= 300:
        return entry["output"]
    return None


def _usage(prompt: str, text: str) -> dict:
    # Better estimate: ~0.35 tokens per char for mixed content
    p = max(1, int(len(prompt) * 0.35))
    c = max(1, int(len(text or "") * 0.35))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _upload_images(images: list) -> list:
    """Upload images and return list of file references. Returns None if no images."""
    if not images:
        return None
    file_refs = []
    for item in images:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                data, mime = item
                if isinstance(data, str):
                    data = fetch_image_bytes(data)
                    mime = mime or "image/png"
                if data:
                    ref = upload_image(data, "image.png", mime or "image/png")
                    if ref:
                        file_refs.append(ref)
        except Exception as e:
            log(f"Image upload failed: {e}")
    return file_refs if file_refs else None


class GeminiHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log(fmt % args)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _parse_body(self, body: bytes) -> dict:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        auth = self.headers.get("Authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else self.headers.get("x-api-key", "")
        return key in keys

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            if self.path.startswith("/v1/") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google",
                     "description": c["desc"],
                     "supports_image_input": True}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
                     "inputImageSupported": True}
                    for n, c in MODELS.items()
                ]})
            elif self.path == "/":
                self.send_json({"status": "ok", "version": __version__, "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            if self.path.startswith("/v1/") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            elif self.path == "/v1/responses":
                self._handle_responses(body)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    # ─── /v1/chat/completions ─────────────────────────────────────────────────

    def _handle_chat(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        prompt, images = messages_to_prompt(req.get("messages", []), tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream and (not tools or tool_choice == "none"):
            try:
                self._start_sse()
                for delta in generate_stream(prompt, model_id, think_mode, _upload_images(images), extra_fields):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                end = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            text = generate(prompt, model_id, think_mode, _upload_images(images), extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text, tools)
            # Retry: if tools expected but parse returned nothing, re-prompt
            retries = 0
            while not tool_calls and text and retries < 2:
                retry_prompt = prompt + "\n\n[System]: You MUST call one of the available tools. Output only a ```tool_call block with valid JSON. Do not respond with text."
                try:
                    text = generate(retry_prompt, model_id, think_mode, _upload_images(images), extra_fields)
                except Exception:
                    break
                if text:
                    text, tool_calls = parse_tool_calls(text, tools)
                else:
                    break
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            self._start_sse()
            text_content = msg.get("content")
            tc_list = msg.get("tool_calls")
            if text_content:
                delta = {"role": "assistant", "content": text_content}
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            if tc_list:
                delta = {"tool_calls": tc_list}
                if not text_content:
                    delta["role"] = "assistant"
                    delta["content"] = None
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            end = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                   "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": _usage(prompt, text),
            })

    # ─── /v1/responses (Codex CLI / OpenCode) ─────────────────────────────────

    def _handle_responses(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        model_name, model_id, think_mode, err, extra_fields = resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        input_items = req.get("input", [])
        tools = req.get("tools")
        messages = []

        # Multi-turn: handle previous_response_id
        prev_rid = req.get("previous_response_id")
        if prev_rid:
            prev_output = _get_cached_response(prev_rid)
            if prev_output:
                for item in prev_output:
                    if item.get("type") == "message":
                        text = ""
                        for cp in item.get("content", []):
                            if cp.get("type") == "output_text":
                                text += cp.get("text", "")
                        if text:
                            messages.append({"role": "assistant", "content": text})
                    elif item.get("type") == "function_call":
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": item.get("id", ""),
                                "type": "function",
                                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")}
                            }]
                        })

        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("type") == "input_image":
                        # Top-level image in input array (Responses API format)
                        messages.append({"role": "user", "content": [item]})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text": text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call": tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        content = item.get("content", "")
                        # Pass content list as-is so image items reach messages_to_prompt
                        messages.append({"role": role, "content": content})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        tool_choice = req.get("tool_choice", "auto")
        prompt, images = messages_to_prompt(messages, tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        file_refs = _upload_images(images)
        stream = req.get("stream", False)
        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        has_tools = bool(tools) and tool_choice != "none"

        # Streaming with delta events (no tool calls) — real-time for Codex
        if stream and not has_tools:
            try:
                self._start_sse()
                now = int(time.time())
                ev = {"type": "response.created", "response": {"id": rid, "object": "response", "created_at": now, "status": "in_progress", "model": model_name, "output": []}}
                self.wfile.write(f"event: response.created\ndata: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

                full_text = ""
                try:
                    for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                        if not delta:
                            continue
                        full_text += delta
                        ev = {"type": "response.output_text.delta", "item_id": mid, "content_index": 0, "delta": delta}
                        self.wfile.write(f"event: response.output_text.delta\ndata: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except Exception as stream_err:
                    log(f"Response streaming error: {stream_err}")
                    # Send partial output with error status
                    output = []
                    if full_text:
                        output.append({"type": "message", "id": mid, "role": "assistant", "status": "incomplete",
                                       "content": [{"type": "output_text", "text": full_text, "annotations": []}]})
                    _cache_response(rid, output)
                    err_ev = {"type": "response.error", "code": "stream_error", "message": str(stream_err)}
                    self.wfile.write(f"event: response.error\ndata: {json.dumps(err_ev)}\n\n".encode())
                    resp_obj = {"id": rid, "object": "response", "created_at": now, "status": "failed", "model": model_name, "output": output,
                                "usage": _usage(prompt, full_text)}
                    self.wfile.write(f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': resp_obj})}\n\n".encode())
                    self.wfile.flush()
                    return

                output = [{"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": full_text, "annotations": []}]}]
                _cache_response(rid, output)
                for ci, cp in enumerate(output[0]["content"]):
                    ev = {"type": "response.output_text.done", "item_id": mid, "content_index": ci, "text": cp["text"]}
                    self.wfile.write(f"event: response.output_text.done\ndata: {json.dumps(ev)}\n\n".encode())
                resp_obj = {"id": rid, "object": "response", "created_at": now, "status": "completed", "model": model_name, "output": output,
                            "usage": _usage(prompt, full_text)}
                self.wfile.write(f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': resp_obj})}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # Non-streaming (or with tool calls — need full parse)
        try:
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        tool_calls = None
        if has_tools and text:
            text, tool_calls = parse_tool_calls(text, tools)

        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})

        _cache_response(rid, output)

        now = int(time.time())
        if stream:
            self._start_sse()
            ev = {"type": "response.created", "response": {"id": rid, "object": "response", "created_at": now, "status": "in_progress", "model": model_name, "output": []}}
            self.wfile.write(f"event: response.created\ndata: {json.dumps(ev)}\n\n".encode())
            for item in output:
                if item["type"] == "function_call":
                    ev = {"type": "response.function_call_arguments.delta", "item_id": item["id"], "call_id": item["call_id"], "delta": item["arguments"]}
                    self.wfile.write(f"event: response.function_call_arguments.delta\ndata: {json.dumps(ev)}\n\n".encode())
                    ev = {"type": "response.function_call_arguments.done", "item_id": item["id"], "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
                    self.wfile.write(f"event: response.function_call_arguments.done\ndata: {json.dumps(ev)}\n\n".encode())
                elif item["type"] == "message":
                    for ci, cp in enumerate(item["content"]):
                        if cp.get("text"):
                            ev = {"type": "response.output_text.delta", "item_id": item["id"], "content_index": ci, "delta": cp["text"]}
                            self.wfile.write(f"event: response.output_text.delta\ndata: {json.dumps(ev)}\n\n".encode())
                        ev = {"type": "response.output_text.done", "item_id": item["id"], "content_index": ci, "text": cp["text"]}
                        self.wfile.write(f"event: response.output_text.done\ndata: {json.dumps(ev)}\n\n".encode())
            resp_obj = {"id": rid, "object": "response", "created_at": now, "status": "completed", "model": model_name, "output": output,
                        "usage": _usage(prompt, text or "")}
            self.wfile.write(f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': resp_obj})}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json({"id": rid, "object": "response", "created_at": now, "status": "completed",
                            "model": model_name, "output": output,
                            "usage": _usage(prompt, text or "")})

    # ─── /v1beta/models (Google Gemini CLI) ──────────────────────────────────

    def _handle_google_generate(self, body: bytes, stream: bool):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        model_name = m.group(1) if m else CONFIG["default_model"]
        model_name, model_id, think_mode, err, extra_fields = resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tool_config = req.get("toolConfig", {})
        fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")
        has_tools = bool(req.get("tools")) and fc_mode != "NONE"
        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        file_refs = _upload_images(images)
        log(f"Google API: model={model_name} stream={stream} tools={has_tools} prompt_len={len(prompt)}")

        if stream and not has_tools:
            try:
                self._start_sse()
                full_text = ""
                for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
                    if not delta:
                        continue
                    full_text += delta
                    chunk_obj = {
                        "candidates": [{"content": {"parts": [{"text": delta}], "role": "model"}, "index": 0}],
                        "modelVersion": model_name,
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": max(1, int(len(prompt) * 0.35)),
                        "candidatesTokenCount": max(1, int(len(full_text) * 0.35)),
                        "totalTokenCount": max(1, int(len(prompt) * 0.35)) + max(1, int(len(full_text) * 0.35)),
                    },
                    "modelVersion": model_name,
                }
                self.wfile.write(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if not text:
            log("Warning: empty response from Gemini")

        response_parts = []
        if has_tools and text:
            clean_text, function_calls = parse_google_function_calls(text)
            if function_calls:
                if clean_text:
                    response_parts.append({"text": clean_text})
                for fc in function_calls:
                    response_parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
            else:
                response_parts.append({"text": text})
        else:
            response_parts.append({"text": text or "I apologize, but I was unable to generate a response. Please try again."})

        candidate = {
            "content": {"parts": response_parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": max(1, int(len(prompt) * 0.35)),
            "candidatesTokenCount": max(1, int(len(text or "") * 0.35)),
            "totalTokenCount": max(1, int(len(prompt) * 0.35)) + max(1, int(len(text or "") * 0.35)),
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self._start_sse()
            self.wfile.write(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
