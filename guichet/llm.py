"""One way to call a language model, whoever serves it.

    model = "claude-haiku-5-5"                      → Anthropic API (ANTHROPIC_API_KEY)
    model = "openai/gpt-oss-120b:deepinfra"         → Hugging Face Inference Providers
                                                      (HF_TOKEN; ":provider" pins who serves it)

Both paths take the same system prompt and user message and return the same
`Reply`. For the structured answer form, Claude gets a forced tool call; the
Hugging Face router (OpenAI-compatible) gets the same schema as a forced
function call, falling back to JSON in the text if a provider ignores it.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"


@dataclass
class Reply:
    raw: str                       # what the model returned, for the record
    data: dict = field(default_factory=dict)   # parsed answer form (structured calls only)
    text: str = ""                 # plain text (unstructured calls)
    tokens_in: int = 0
    tokens_out: int = 0
    stop_reason: str = ""


class TransientError(Exception):
    """Rate limit, timeout or server error: worth retrying."""


class AccessError(Exception):
    """Wrong key, missing permission or unknown model: retrying won't help."""


def is_claude(model: str) -> bool:
    return model.startswith("claude")


def safe_name(model: str) -> str:
    """A model id usable in a file name: 'openai/gpt-oss-120b:deepinfra' → 'openai--gpt-oss-120b@deepinfra'."""
    return model.replace("/", "--").replace(":", "@")


# ------------------------------------------------------------------ Anthropic
def _anthropic(client, model, system, user, tool, max_tokens) -> Reply:
    import anthropic

    kwargs: dict = {"max_tokens": max_tokens}
    if tool:
        kwargs |= {"tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]}}
    try:
        message = client.messages.create(model=model, system=system,
                                         messages=[{"role": "user", "content": user}], **kwargs)
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
        raise TransientError(str(exc)) from exc
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as exc:
        raise AccessError(str(exc)) from exc
    reply = Reply(raw="", tokens_in=message.usage.input_tokens, tokens_out=message.usage.output_tokens,
                  stop_reason=getattr(message, "stop_reason", "") or "")
    for block in message.content:
        if getattr(block, "type", "") == "tool_use":
            reply.data = dict(block.input)
            reply.raw = json.dumps(reply.data, ensure_ascii=False)
            return reply
    reply.text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()
    reply.raw = reply.text
    if tool:
        reply.data = _json_from_text(reply.text)
    return reply


# --------------------------------------------------------------- Hugging Face
def _hf(model, system, user, tool, max_tokens, http=None) -> Reply:
    import httpx

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set: put it in .env (never commit it).")
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if tool:
        body["tools"] = [{"type": "function", "function": {
            "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}]
        body["tool_choice"] = {"type": "function", "function": {"name": tool["name"]}}
    client = http or httpx.Client(timeout=180)
    try:
        response = client.post(HF_ROUTER, json=body, headers={"Authorization": f"Bearer {token}"})
    except httpx.TransportError as exc:
        raise TransientError(str(exc)) from exc
    finally:
        if http is None:
            client.close()
    if response.status_code in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code in (401, 402, 403, 404):
        raise AccessError(f"HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    reply = Reply(raw="", tokens_in=usage.get("prompt_tokens", 0), tokens_out=usage.get("completion_tokens", 0),
                  stop_reason=choice.get("finish_reason") or "")
    calls = message.get("tool_calls") or []
    if tool and calls:
        arguments = calls[0].get("function", {}).get("arguments") or "{}"
        reply.raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        reply.data = _json_from_text(reply.raw)
        return reply
    reply.text = (message.get("content") or "").strip()
    reply.raw = reply.text
    if tool:
        reply.data = _json_from_text(reply.text)
    return reply


def _json_from_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return {"found": None, "answer": (text or "").strip(), "citations": [], "parse_error": True}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"found": None, "answer": (text or "").strip(), "citations": [], "parse_error": True}
    if not isinstance(data, dict):
        return {"found": None, "answer": str(data), "citations": [], "parse_error": True}
    data.setdefault("citations", [])
    return data


# -------------------------------------------------------------------- public
def complete(model: str, system: str, user: str, *, tool: dict | None = None, max_tokens: int = 4096,
             client=None, http=None, retries: int = 4) -> Reply:
    """Call the model, retrying transient failures with backoff."""
    for attempt in range(retries):
        try:
            if is_claude(model):
                if client is None:
                    import anthropic

                    client = anthropic.Anthropic()
                return _anthropic(client, model, system, user, tool, max_tokens)
            return _hf(model, system, user, tool, max_tokens, http)
        except TransientError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("unreachable")
