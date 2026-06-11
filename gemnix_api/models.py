"""Model definitions and mapping from Gemini frontend JS source."""

# MODE_CATEGORY enum from 028-6eb337387583.js:
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
    "gemini-3.1-pro-enhanced": {
        "mode": 3, "think": 4, "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
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
    "gemini-2.5-flash": {
        "mode": 7, "think": 4,
        "desc": "Latest fast model",
    },
    "gemini-2.5-pro": {
        "mode": 8, "think": 4,
        "desc": "Latest pro model (needs cookie)",
    },
}

# Model aliases — Codex/OpenCode gửi model ID lạ, map về flash ổn định nhất.
MODEL_ALIASES = {
    "claude-sonnet-4-6": "gemini-3.5-flash",
    "claude-sonnet-4-7": "gemini-3.5-flash",
    "claude-sonnet-4-8": "gemini-3.5-flash",
    "claude-opus-4-8": "gemini-3.5-flash",
    "claude-haiku-4-5": "gemini-3.5-flash",
    "claude-fable-5": "gemini-3.5-flash-thinking",
    "gpt-4o": "gemini-3.5-flash",
    "gpt-4o-mini": "gemini-flash-lite",
    "gpt-4-turbo": "gemini-3.5-flash",
    "deepseek-chat": "gemini-3.5-flash",
    "deepseek-reasoner": "gemini-3.5-flash",
    "claude-opus-4-9": "gemini-3.5-flash-thinking",
    "claude-opus-4-10": "gemini-3.5-flash-thinking",
    "gpt-4.1": "gemini-3.5-flash",
    "gpt-4.1-mini": "gemini-flash-lite",
    "gpt-4.1-nano": "gemini-flash-lite",
    "o4-mini": "gemini-3.5-flash-thinking-lite",
    "gemini-2.5-flash": "gemini-3.5-flash",
    "gemini-2.5-pro": "gemini-3.1-pro",
}

# Prefix-based fallback
_MODEL_PREFIXES = {
    "claude-": "gemini-3.5-flash",
    "gpt-": "gemini-3.5-flash",
    "deepseek-": "gemini-3.5-flash",
}


def resolve_model(model_name: str, default: str = "gemini-3.5-flash"):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields).

    Unknown model names fall back via alias map or to default.
    The returned 'name' is the ORIGINAL requested model (so clients see
    their own model echoed back). The mode/think config comes from the
    mapped Gemini model.
    """
    think_override = None
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None

    original_name = model_name

    # 1. Direct match in MODELS
    cfg = MODELS.get(model_name)
    if not cfg:
        # 2. Check exact alias
        mapped = MODEL_ALIASES.get(model_name)
        if mapped:
            cfg = MODELS.get(mapped)
    if not cfg:
        # 3. Check prefix-based aliases
        for prefix, target in _MODEL_PREFIXES.items():
            if model_name.startswith(prefix) or model_name.lower().startswith(prefix):
                cfg = MODELS.get(target)
                if cfg:
                    from .gemini import log
                    log(f"Alias '{model_name}' -> '{target}' (prefix '{prefix}')")
                    break
    if not cfg:
        from .gemini import log
        log(f"Unknown model '{model_name}', falling back to '{default}'")
        cfg = MODELS[default]

    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = cfg.get("extra")
    return original_name, mode_id, think_mode, None, extra
