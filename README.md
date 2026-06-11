<p align="center">
  <img src="logo.png" width="200" alt="gemnix-api logo">
</p>

# Gemnix API — Gemini Web to OpenAI API Proxy (Premium Edition)

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/lsdefine/gemnix-api)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](https://hub.docker.com/)

Convert Google Gemini's web interface into a full OpenAI-compatible API server. Zero cost, single-file or modular package, optimized for AI coding tools (Codex CLI, Cursor, OpenCode, Cherry Studio).

---

## Quick Start

```bash
pip install httpx
python gemnix_api.py
```

Server starts at `http://localhost:8081/v1`. No configuration needed for anonymous use.

## Configuration

Create `config.json` in the current directory (auto-discovered). Or pass `--config path/to/config.json`.

### config.json reference

```json
{
  "port": 8081,
  "host": "0.0.0.0",
  "retry_attempts": 3,
  "retry_delay_sec": 2,
  "request_delay_sec": 0,
  "request_timeout_sec": 180,
  "gemini_bl": "boq_assistant-bard-web-server_20260525.09_p0",
  "auth_user": null,
  "xsrf_token": null,
  "default_model": "gemini-3.5-flash",
  "log_requests": true,
  "cookie_file": null,
  "proxy": null,
  "proxies": [],
  "api_keys": []
}
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `port` | int | `8081` | HTTP listen port |
| `host` | string | `"0.0.0.0"` | Bind address |
| `retry_attempts` | int | `3` | Number of retries on upstream failure |
| `retry_delay_sec` | int | `2` | Seconds between retries |
| `request_delay_sec` | int | `0` | Delay before each request (rate-limit avoidance) |
| `request_timeout_sec` | int | `180` | Upstream request timeout |
| `gemini_bl` | string | `"boq_assistant-..."` | Gemini build label (update when errors occur) |
| `auth_user` | string/null | `null` | Account index path (`/u/1/` -> `"1"`) |
| `xsrf_token` | string/null | `null` | Page XSRF token (`SNlM0e` from page source) |
| `default_model` | string | `"gemini-3.5-flash"` | Default model when none specified |
| `log_requests` | bool | `true` | Log requests to stderr |
| `cookie_file` | string/null | `null` | Path to cookie file or directory |
| `proxy` | string/null | `null` | Single HTTP proxy URL |
| `proxies` | string[] | `[]` | Multiple proxy URLs (random rotation) |
| `api_keys` | string[] | `[]` | Auth keys; empty = no auth |

**Auth**: When `api_keys` is `[]`, authentication is disabled. When one or more keys are set, endpoints require `Authorization: Bearer <key>` or `x-api-key: <key>`.

**Proxy rotation**: Set `proxies` to a list of URLs and each request picks one at random. Falls back to `proxy` if `proxies` is empty.

### Cookie file for Pro routing

Anonymous access works for all models, but `gemini-3.1-pro` routes to Flash without authentication. To unlock real Pro routing you need a **Gemini Advanced (paid subscription)** account cookie.

**Format** (single-line):
```
SID=xxx; HSID=xxx; SSID=xxx; APISID=xxx; SAPISID=xxx; __Secure-1PSID=xxx
```

Or JSON:
```json
{"cookie": "SID=xxx; HSID=xxx; ...", "sapisid": "your_sapisid_value"}
```

**Getting cookies**: Chrome -> DevTools (F12) -> Application -> Cookies -> `https://gemini.google.com`. Copy `SID`, `HSID`, `SSID`, `APISID`, `SAPISID`, `__Secure-1PSID`.

**Cookie directory**: Set `cookie_file` to a directory path containing `.txt` cookie files. A random one is picked per request — useful for multi-account rotation.

**Auth user and XSRF**: If your browser URL has `/u/1/` in the path, set `auth_user` to `"1"`. The XSRF token (`SNlM0e`) is found in the Gemini page HTML source.

## Client Setup

### Codex CLI

```bash
export OPENAI_BASE_URL=http://localhost:8081/v1
export OPENAI_API_KEY=sk-your-key   # omit if api_keys is empty
export OPENAI_MODEL=gemini-3.5-flash-thinking
```

Codex CLI uses the **Responses API** (`/v1/responses`) — fully supported.

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")
resp = client.chat.completions.create(
    model="gemini-3.5-flash-thinking",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(resp.choices[0].message.content)
```

### curl

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

### Gemini CLI

```bash
export GEMINI_API_KEY=none
export GOOGLE_GEMINI_BASE_URL=http://localhost:8081
gemini
```

Supports Google native API:
- `GET /v1beta/models` — list models
- `POST /v1beta/models/{model}:generateContent` — non-streaming
- `POST /v1beta/models/{model}:streamGenerateContent` — streaming (SSE)

### Cherry Studio / ChatBox / any OpenAI client

| Field | Value |
|-------|-------|
| Base URL | `http://localhost:8081/v1` |
| API Key | any `api_keys` value from `config.json`; anything if auth is off |
| Model | `gemini-3.5-flash-thinking` |

## Models

| Model | Mode | Description | Output |
|-------|------|-------------|--------|
| `gemini-3.5-flash` | 1 | Fast general-purpose | ~12k chars |
| `gemini-3.5-flash-thinking` | 2 | Deep thinking, longest output | ~20k chars |
| `gemini-3.1-pro` | 3 | Pro (needs cookie for real routing) | ~12k chars |
| `gemini-auto` | 4 | Auto model selection | varies |
| `gemini-3.5-flash-thinking-lite` | 5 | Adaptive thinking depth | ~15k chars |
| `gemini-flash-lite` | 6 | Lightweight fast | ~10k chars |

### Thinking override

Append `@think=N` to any model to override thinking depth (0=deepest, 4=shallowest):

```
gemini-3.5-flash-thinking@think=0   # deepest (default)
gemini-3.5-flash-thinking@think=2   # medium
gemini-3.5-flash-thinking@think=4   # shallowest
```

### Model alias map

The server exposes models at `/v1/models` in OpenAI format. You can configure any client to use the model IDs in the table above.

## Tool Calling

Full OpenAI-style function calling support. Tools are injected as system instructions; the model responds with `tool_call` code blocks that the server parses into OpenAI-formatted `tool_calls`.

```python
client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")

resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }],
    tool_choice="auto"
)

if resp.choices[0].message.tool_calls:
    for tc in resp.choices[0].message.tool_calls:
        print(f"Call: {tc.function.name}({tc.function.arguments})")
```

Supports:
- `tool_choice: "auto"` — model decides
- `tool_choice: "required"` — must call at least one tool
- `tool_choice: "none"` — suppress all tool calls
- `tool_choice: {"type": "function", "function": {"name": "..."}}` — force specific tool

## Responses API (Codex CLI)

Full OpenAI Responses API (`/v1/responses`) with streaming support, used by Codex CLI.

**Non-streaming**:
```bash
curl http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "gemini-3.5-flash-thinking",
    "input": "Write a Python script to list files",
    "instructions": "You are a helpful coding assistant."
  }'
```

**Streaming with SSE events**:
```bash
curl -N http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "gemini-3.5-flash-thinking",
    "input": "Write a Python script",
    "stream": true
  }'
```

Events: `response.created`, `response.output_text.delta`, `response.output_text.done`, `response.function_call_arguments.done`, `response.completed`, `response.error`.

**Multi-turn** via `previous_response_id` — the server caches prior response output for 5 minutes.

## Docker

```bash
cp config.example.json config.json
# Edit config.json as needed
docker build -t gemnix-api .
docker run -d --name gemnix-api \
  -p 8081:8081 \
  -v ./config.json:/app/config.json \
  gemnix-api
```

With cookie file:
```bash
docker run -d --name gemnix-api \
  -p 8081:8081 \
  -v ./config.json:/app/config.json \
  -v ./cookie.txt:/app/cookie.txt \
  gemnix-api
```

Set `"cookie_file": "/app/cookie.txt"` in your config.json.

### Docker Compose

```yaml
services:
  gemnix-api:
    build: .
    container_name: gemnix-api
    network_mode: host
    ports:
      - "8081:8081"
    volumes:
      - ./config.json:/app/config.json
      - ./cookies:/app/cookies
    restart: unless-stopped
```

### Network note

If you get empty responses (`content: null`) with Docker's default bridge network, switch to host networking. Gemini's upstream may reject requests from certain Docker NAT IP ranges.

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check + version info |
| `GET` | `/v1/models` | List models (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat completions (OpenAI format) |
| `POST` | `/v1/responses` | Responses API (Codex CLI format) |
| `GET` | `/v1beta/models` | List models (Google format) |
| `POST` | `/v1beta/models/{model}:generateContent` | Generate (Google format) |
| `POST` | `/v1beta/models/{model}:streamGenerateContent` | Stream generate (Google format) |

### Authentication

- `api_keys: []` — no authentication required
- `api_keys: ["sk-xxx"]` — require `Authorization: Bearer sk-xxx` or `x-api-key: sk-xxx`

### Streaming

Chat completions stream SSE when `"stream": true`. Tool calls in streaming mode are delivered as a single chunk (full parse required).

Responses API streaming delivers granular SSE events for progressive rendering.

### Error handling

Errors are returned as OpenAI-format error objects:

```json
{"error": {"message": "upstream error: BardErrorInfo [5]"}}
```

The server retries on upstream failures according to `retry_attempts` and `retry_delay_sec`. If all retries fail, a `502` is returned with the error detail.

### Limitations

- **No image/multimodal input**: Gemini's image upload uses a proprietary streaming RPC protocol. Image inputs are ignored.
- **Not real Pro/Ultra**: Without a Gemini Advanced cookie, `gemini-3.1-pro` routes to Flash. The model selection is a UI preference, not a backend switch.
- **Single-turn per request**: Each request is independent. Multi-turn context is simulated by including previous messages in the prompt.
- **Rate limits**: Google may throttle high-frequency requests. Use `request_delay_sec` to add a delay between requests.

## Requirements

- Python 3.8+
- `httpx` (`pip install httpx`) — required for streaming
- Network access to `gemini.google.com`

## How It Works

This tool reverse-engineers Google Gemini's web StreamGenerate protocol. Requests are sent to the same endpoint Gemini web uses, converting between OpenAI API format and Gemini's internal protobuf-like format. Model selection is controlled by field `[79]` in the request payload, mapped from Gemini's `MODE_CATEGORY` enum.

## License

MIT
