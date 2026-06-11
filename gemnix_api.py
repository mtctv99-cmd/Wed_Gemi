#!/usr/bin/env python3
"""
gemnix-api - Gemini Web to OpenAI API proxy — Premium Edition.

Converts Google Gemini's web interface into an OpenAI-compatible API server.
Optimized for AI coding tools (Codex CLI, OpenCode, Cursor).

Usage:
    pip install httpx
    python gemnix_api.py [--port 8081] [--config config.json]

Client configuration:
    Base URL: http://localhost:8081/v1
    API Key: from config.json api_keys (or anything if empty)
"""
import json
import urllib.request
import urllib.parse
import time
import ssl
import sys
import uuid
import re
import os
import hashlib
import argparse
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

try:
    from curl_cffi import requests as cffi_requests
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

__version__ = "2.0.0"

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_delay_sec": 0,  # No delay
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260525.09_p0",
    "auth_user": None,
    "xsrf_token": None,
    "default_model": "gemini-3.5-flash",
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,           # single proxy (fallback)
    "proxies": [],           # list of proxies for rotation: ["http://x:port", ...]
    "api_keys": [],
}

CONFIG = dict(DEFAULT_CONFIG)

# ─── Models ──────────────────────────────────────────────────────────────────
# Mapping from JS source: MODE_CATEGORY enum (028-6eb337387583.js)
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

MODELS = {
    "gemini-3.5-flash": {
        "mode": 1, "think": 4,
        "desc": "Fast general-purpose model",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4,
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-auto": {
        "mode": 4, "think": 4,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4,
        "desc": "Lightweight fast model",
    },
}

# ─── Utilities ───────────────────────────────────────────────────────────────

def log(msg: str):
    if CONFIG["log_requests"]:
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


import random

def get_cookie_files() -> list:
    """Get list of cookie files from cookie_file config (can be file or directory)."""
    cookie_path = CONFIG.get("cookie_file")
    if not cookie_path or not os.path.exists(cookie_path):
        return []

    if os.path.isdir(cookie_path):
        files = []
        for f in os.listdir(cookie_path):
            if f.endswith(".txt"):
                files.append(os.path.join(cookie_path, f))
        return files
    return [cookie_path]

def load_cookie() -> tuple:
    """Load a random cookie from available files. Returns (cookie_str, sapisid)."""
    files = get_cookie_files()
    if not files:
        return "", None

    # Pick a random cookie file (round-robin could be implemented, but random is simpler for concurrency)
    cookie_file = random.choice(files)

    try:
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error from {cookie_file}: {e}")
        return "", None


def get_proxy() -> str:
    """Get a random proxy from config list, or single proxy if defined."""
    proxies = CONFIG.get("proxies", [])
    if proxies and isinstance(proxies, list) and len(proxies) > 0:
        return random.choice(proxies)
    return CONFIG.get("proxy")

def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


# ─── Responses API Cache ──────────────────────────────────────────────────────

_response_cache: dict = {}

def _cache_response(rid: str, output: list):
    _response_cache[rid] = {"output": output, "ts": time.time()}
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
    p = max(1, int(len(prompt) * 0.35))
    c = max(1, int(len(text or "") * 0.35))
    return {"input_tokens": p, "output_tokens": c, "total_tokens": p + c}


# ─── Gemini Protocol ─────────────────────────────────────────────────────────

def gemini_stream_generate(prompt: str, model_id: int, think_mode: int) -> str:
    """Send prompt to Gemini StreamGenerate with retry."""
    inner = [None] * 80
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params).encode()
    reqid = int(time.time()) % 1000000
    prefix = account_prefix()
    url = (
        f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])

    cookie_str, sapisid = load_cookie()
    last_err = None
    # Try with cookie first if available, then fallback to anonymous if it fails
    use_cookie_options = [True, False] if cookie_str else [False]

    for use_cookie in use_cookie_options:
        for attempt in range(CONFIG["retry_attempts"]):
            try:
                # Add delay before request to prevent rate limits
                delay = CONFIG.get("request_delay_sec", 0)
                if delay > 0:
                    time.sleep(delay)

                current_headers = headers.copy()
                if use_cookie:
                    current_headers["Cookie"] = cookie_str
                    if sapisid:
                        current_headers["Authorization"] = make_sapisidhash(sapisid)

                proxy = get_proxy()
                if HAS_CFFI:
                    proxies = {"http": proxy, "https": proxy} if proxy else None
                    resp = cffi_requests.post(
                        url,
                        data=body,
                        headers=current_headers,
                        proxies=proxies,
                        timeout=CONFIG["request_timeout_sec"],
                        impersonate="chrome120"
                    )
                    return resp.text
                else:
                    req = urllib.request.Request(url, data=body, headers=current_headers, method="POST")
                    ctx = ssl.create_default_context()
                    if proxy:
                        opener = urllib.request.build_opener(
                            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                            urllib.request.HTTPSHandler(context=ctx)
                        )
                        resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
                    else:
                        resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
                    return resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                last_err = e
                if attempt < CONFIG["retry_attempts"] - 1:
                    log(f"Retry {attempt+1}/{CONFIG['retry_attempts']} (cookie={use_cookie}): {e}")
                    time.sleep(CONFIG["retry_delay_sec"])
                elif use_cookie:
                    log(f"Cookie request failed, falling back to anonymous mode: {e}")
                    break
    raise last_err


def gemini_stream_generate_iter(prompt: str, model_id: int, think_mode: int):
    """Send prompt and yield incremental text deltas using httpx streaming."""
    inner = [None] * 80
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id

    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    body = urllib.parse.urlencode(params)
    reqid = int(time.time()) % 1000000
    prefix = account_prefix()
    url = (
        f"https://gemini.google.com{prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    proxy = get_proxy()

    def get_stream_iter(use_cookie):
        nonlocal prev_text

        # Add delay before request to prevent rate limits
        delay = CONFIG.get("request_delay_sec", 0)
        if delay > 0:
            time.sleep(delay)

        current_headers = headers.copy()
        if use_cookie:
            current_headers["Cookie"] = cookie_str
            if sapisid:
                current_headers["Authorization"] = make_sapisidhash(sapisid)

        if HAS_CFFI:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            with cffi_requests.Session() as s:
                resp = s.post(
                    url,
                    data=body,
                    headers=current_headers,
                    proxies=proxies,
                    timeout=CONFIG["request_timeout_sec"],
                    impersonate="chrome120",
                    stream=True
                )
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        line = chunk.decode("utf-8", errors="replace")
                        if '"wrb.fr"' not in line or len(line) < 200:
                            continue
                        try:
                            arr = json.loads(line)
                            inner_str = arr[0][2]
                            if not inner_str or len(inner_str) < 50:
                                continue
                            inner2 = json.loads(inner_str)
                            if isinstance(inner2, list) and len(inner2) > 4 and inner2[4]:
                                for part in inner2[4]:
                                    if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                                        for t in part[1]:
                                            if isinstance(t, str) and len(t) > len(prev_text):
                                                delta = t[len(prev_text):]
                                                delta = clean_gemini_text(delta)
                                                if delta:
                                                    yield delta
                                                prev_text = t
                        except (json.JSONDecodeError, IndexError, TypeError):
                            pass
            return

        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        with httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True) as client:
            with client.stream("POST", url, content=body, headers=current_headers) as resp:
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        import re as _re
                        m = _re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if m:
                            raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{m.group(1)}]")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if '"wrb.fr"' not in line or len(line) < 200:
                            continue
                        try:
                            arr = json.loads(line)
                            inner_str = arr[0][2]
                            if not inner_str or len(inner_str) < 50:
                                continue
                            inner2 = json.loads(inner_str)
                            if isinstance(inner2, list) and len(inner2) > 4 and inner2[4]:
                                for part in inner2[4]:
                                    if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                                        for t in part[1]:
                                            if isinstance(t, str) and len(t) > len(prev_text):
                                                delta = t[len(prev_text):]
                                                delta = clean_gemini_text(delta)
                                                if delta:
                                                    yield delta
                                                prev_text = t
                        except (json.JSONDecodeError, IndexError, TypeError):
                            pass

    if not HAS_HTTPX:
        # Fallback: non-streaming with urllib (already has fallback logic)
        raw = gemini_stream_generate(prompt, model_id, think_mode)
        text = extract_response_text(raw)
        if text:
            yield text
        return

    prev_text = ""
    try:
        if cookie_str:
            for delta in get_stream_iter(True):
                yield delta
        else:
            for delta in get_stream_iter(False):
                yield delta
    except Exception as e:
        log(f"Stream with cookie failed, falling back to anonymous: {e}")
        if cookie_str:
            prev_text = "" # Reset for anonymous retry
            for delta in get_stream_iter(False):
                yield delta
        else:
            raise e


def clean_gemini_text(text: str) -> str:
    """Remove internal code execution artifacts."""
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    return text.strip()


def extract_response_text(raw: str) -> str:
    """Parse StreamGenerate response to extract final text."""
    import re as _re
    bard_err = _re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")
    texts = []
    for line in raw.split("\n"):
        if '"wrb.fr"' not in line or len(line) < 200:
            continue
        try:
            arr = json.loads(line)
            inner_str = arr[0][2]
            if not inner_str or len(inner_str) < 50:
                continue
            inner = json.loads(inner_str)
            if isinstance(inner, list) and len(inner) > 4 and inner[4]:
                for part in inner[4]:
                    if isinstance(part, list) and len(part) > 1 and part[1]:
                        if isinstance(part[1], list):
                            for t in part[1]:
                                if isinstance(t, str) and len(t) > 0:
                                    texts.append(t)
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    text = ""
    for t in reversed(texts):
        if t.strip():
            text = t
            break
    return clean_gemini_text(text)


# ─── OpenAI Format Helpers ───────────────────────────────────────────────────

def messages_to_prompt(messages: list, tools: list = None, tool_choice=None) -> str:
    """Convert OpenAI messages to prompt string."""
    parts = []
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
            constraint = ""
            if tool_choice == "required":
                constraint = "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
            elif isinstance(tool_choice, dict):
                fn_name = tool_choice.get("function", {}).get("name", "")
                if fn_name:
                    constraint = f'\n\nIMPORTANT: You MUST call the tool "{fn_name}". Do not call other tools.'
            parts.append(
                "[System instruction]: You have access to tools. "
                "To call a tool, respond with:\n"
                '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
                "Only use tool_call blocks when needed.\n\n"
                f"Available tools:\n{json.dumps(tool_defs, indent=2)}{constraint}"
            )
    if tool_choice == "none" and tools:
        parts.append("[System instruction]: Do NOT call any tools. Respond with text only.")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if c.get("type") in ("text", "input_text")
            )
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
                parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(tc_strs))
            else:
                parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('name', '')}]: {content}")
        else:
            parts.append(content if content else "")
    return "\n\n".join(p for p in parts if p)


def parse_tool_calls(text: str) -> tuple:
    """Extract tool_call blocks. Returns (clean_text, tool_calls_list)."""
    tool_calls = []
    pattern = r'```tool_call\s*\n(.*?)\n```'
    for match in re.findall(pattern, text, re.DOTALL):
        try:
            data = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False),
                },
            })
        except (json.JSONDecodeError, KeyError):
            pass
    clean = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return clean, tool_calls


# ─── HTTP Handler ────────────────────────────────────────────────────────────

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
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in MODELS.items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self._handle_google_models_list()
            elif self.path == "/":
                self.send_json({"status": "ok", "version": __version__,
                                "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"GET error: {e}")

    def do_POST(self):
        try:
            if self.path.startswith("/v1/") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            if self.path == "/v1/chat/completions":
                self.handle_chat(body)
            elif self.path == "/v1/responses":
                self.handle_responses(body)
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

    def _resolve_model(self, model_name):
        think_override = None
        if "@think=" in model_name:
            model_name, think_str = model_name.rsplit("@think=", 1)
            think_override = int(think_str)
        cfg = MODELS.get(model_name)
        if not cfg:
            return None, None, None, f"Unknown model: {model_name}"
        return model_name, cfg["mode"], (think_override if think_override is not None else cfg["think"]), None

    def _call_gemini(self, prompt, model_id, think_mode, tools):
        raw = gemini_stream_generate(prompt, model_id, think_mode)
        text = extract_response_text(raw)
        tool_calls = None
        if tools and text:
            text, tool_calls = parse_tool_calls(text)
        return text or "", tool_calls

    def handle_chat(self, body: bytes):
        req = json.loads(body)
        model_name, model_id, think_mode, err = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        prompt = messages_to_prompt(req.get("messages", []), tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream and not tools:
            # True streaming: forward chunks as they arrive
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                for delta_text in gemini_stream_generate_iter(prompt, model_id, think_mode):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                # Final chunk
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
            return

        # Non-streaming (or tool calling which needs full response)
        try:
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, tools)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            # Stream mode with tools: send as single chunk (need full parse for tool_calls)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text)//4,
                          "total_tokens": (len(prompt)+len(text))//4},
            })

    def handle_responses(self, body: bytes):
        """OpenAI Responses API for Codex CLI compatibility."""
        req = json.loads(body)
        model_name, model_id, think_mode, err = self._resolve_model(
            req.get("model", CONFIG["default_model"]))
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        input_items = req.get("input", [])
        raw_tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        has_tools = bool(raw_tools) and tool_choice != "none"
        tools_for_prompt = raw_tools if has_tools else None

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
                        if isinstance(content, list):
                            content = " ".join(c.get("text", "") for c in content if c.get("type") in ("text", "input_text"))
                        messages.append({"role": role, "content": content})

        if raw_tools:
            raw_tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                         if t.get("type") == "function" and "function" not in t else t for t in raw_tools]

        prompt = messages_to_prompt(messages, tools_for_prompt, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        stream = req.get("stream", False)

        # Streaming with delta events (no tool calls)
        if stream and not has_tools:
            try:
                self._start_sse()
                now = int(time.time())
                ev = {"type": "response.created", "response": {"id": rid, "object": "response", "created_at": now, "status": "in_progress", "model": model_name, "output": []}}
                self.wfile.write(f"event: response.created\ndata: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

                full_text = ""
                try:
                    for delta in gemini_stream_generate_iter(prompt, model_id, think_mode):
                        if not delta:
                            continue
                        full_text += delta
                        ev = {"type": "response.output_text.delta", "item_id": mid, "content_index": 0, "delta": delta}
                        self.wfile.write(f"event: response.output_text.delta\ndata: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except Exception as stream_err:
                    log(f"Response streaming error: {stream_err}")
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
            text, tool_calls = self._call_gemini(prompt, model_id, think_mode, raw_tools if has_tools else None)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

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
                    ev = {"type": "response.function_call_arguments.done", "item_id": item["id"], "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
                    self.wfile.write(f"event: response.function_call_arguments.done\ndata: {json.dumps(ev)}\n\n".encode())
                elif item["type"] == "message":
                    for ci, cp in enumerate(item["content"]):
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


    # ─── Google Native API (Gemini CLI compatible) ────────────────────────────

    def _parse_google_model_from_path(self):
        """Extract model name from /v1beta/models/{model}:method path."""
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        if m:
            return m.group(1)
        return None

    def _handle_google_models_list(self):
        """GET /v1beta/models — Google AI format model list."""
        models = []
        for name, cfg in MODELS.items():
            models.append({
                "name": f"models/{name}",
                "displayName": name,
                "description": cfg["desc"],
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
            })
        self.send_json({"models": models})

    def _google_contents_to_prompt(self, req: dict) -> str:
        """Convert Google API contents format to prompt string."""
        parts = []
        sys_inst = req.get("systemInstruction")
        if sys_inst:
            sys_parts = sys_inst.get("parts", [])
            sys_text = " ".join(p.get("text", "") for p in sys_parts if p.get("text"))
            if sys_text:
                parts.append(f"[System instruction]: {sys_text}")

        for content in req.get("contents", []):
            role = content.get("role", "user")
            text_parts = []
            for p in content.get("parts", []):
                if p.get("text"):
                    text_parts.append(p["text"])
            text = " ".join(text_parts)
            if role == "model":
                parts.append(f"[Assistant]: {text}")
            else:
                parts.append(text)
        return "\n\n".join(p for p in parts if p)

    def _handle_google_generate(self, body: bytes, stream: bool):
        """Handle Google native generateContent / streamGenerateContent."""
        req = json.loads(body)
        model_name = self._parse_google_model_from_path()
        if not model_name:
            self.send_json({"error": {"message": "model not specified in path"}}, 400)
            return

        model_name, model_id, think_mode, err = self._resolve_model(model_name)
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        prompt = self._google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        try:
            text, _ = self._call_gemini(prompt, model_id, think_mode, None)
        except Exception as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        candidate = {
            "content": {"parts": [{"text": text or ""}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text) // 4,
            "totalTokenCount": (len(prompt) + len(text)) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps(response_obj)}\n\n".encode())
            self.wfile.flush()
        else:
            self.send_json(response_obj)


# ─── Main ────────────────────────────────────────────────────────────────────

def load_config(path: str):
    if path and os.path.exists(path):
        with open(path) as f:
            CONFIG.update(json.load(f))
        log(f"Config loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Web to OpenAI API")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cookie-file", type=str, default=None, help="Path to cookie file")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--version", action="version", version=f"gemnix-api {__version__}")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("GEMINI_WEB2API_CONFIG")
    if not config_path:
        for p in ["./config.json", os.path.expanduser("~/.config/gemnix-api/config.json")]:
            if os.path.exists(p):
                config_path = p
                break
    load_config(config_path)

    if args.port:
        CONFIG["port"] = args.port
    if args.cookie_file:
        CONFIG["cookie_file"] = args.cookie_file
    if args.proxy:
        CONFIG["proxy"] = args.proxy

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    port = CONFIG["port"]
    server = ThreadedServer((CONFIG["host"], port), GeminiHandler)
    print(f"gemnix-api v{__version__}")
    print(f"  Listening: http://0.0.0.0:{port}")
    print(f"  Base URL:  http://localhost:{port}/v1")
    print(f"  Models:    {', '.join(MODELS.keys())}")
    print(f"  Cookie:    {'yes (' + CONFIG['cookie_file'] + ')' if CONFIG.get('cookie_file') else 'none (anonymous)'}")
    print(f"  Proxy:     {CONFIG.get('proxy') or 'none (uses system env HTTP_PROXY/HTTPS_PROXY)'}")
    print(f"  Retry:     {CONFIG['retry_attempts']}x / {CONFIG['retry_delay_sec']}s")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
