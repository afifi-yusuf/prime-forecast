"""Bedrock Converse client for the BLF leak filter + summarizer.

Ported from haruspex agent/blf/llm_client.py, adapted for async environments:
boto3 is synchronous, so every call is offloaded via asyncio.to_thread to keep
the verifiers event loop unblocked.

Env vars:
  PF_LLM_FILTER          — "1"/"true" enables the LLM filter+summarizer (default on
                           when Bedrock credentials are present, off otherwise)
  BEDROCK_API_KEY        — short/long-term Bedrock API key (preferred; mapped to
                           AWS_BEARER_TOKEN_BEDROCK for boto3)
  AWS_BEARER_TOKEN_BEDROCK — official boto3 bearer-token env var (also accepted)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION — IAM fallback
  PF_FILTER_MODEL        — filter model id (default Claude Haiku 4.5)
  PF_SUMMARIZE_MODEL     — summarizer model id (default Nova 2 Lite)

Prefer a *long-term* Bedrock API key for Hosted Training. Short-term keys encode
X-Amz-Expires (often 12h) and will start returning AccessDenied mid-run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager

_DEFAULT_REGION = "us-east-1"
_DEFAULT_FILTER = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_SUMMARIZE = "us.amazon.nova-2-lite-v1:0"

_log = logging.getLogger(__name__)
_client = None


class BedrockAuthError(RuntimeError):
    """Raised when Bedrock rejects the API key / IAM credentials."""


def _bedrock_api_key() -> str:
    """Return a Bedrock bearer API key if configured."""
    return (
        os.environ.get("BEDROCK_API_KEY", "").strip()
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    )


def _has_iam_creds() -> bool:
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
        or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
    )


def _drop_bearer_env() -> None:
    """Remove bearer-token env vars so boto3 falls back to the IAM chain.

    boto3 prefers AWS_BEARER_TOKEN_BEDROCK over IAM keys whenever it is set —
    a stale/expired token (e.g. injected by dashboard-stored secrets) would
    otherwise shadow perfectly valid IAM credentials for the whole run.
    """
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"):
        os.environ.pop(name, None)


def _ensure_bearer_env() -> None:
    """Map BEDROCK_API_KEY → AWS_BEARER_TOKEN_BEDROCK so boto3 picks it up."""
    key = _bedrock_api_key()
    if key and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key


def llm_enabled() -> bool:
    raw = os.environ.get("PF_LLM_FILTER", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    # Auto: enabled when a Bedrock API key or IAM credentials are present.
    return bool(
        _bedrock_api_key()
        or os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("AWS_PROFILE")
    )


def bedrock_region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or _DEFAULT_REGION


def filter_model() -> str:
    return os.environ.get("PF_FILTER_MODEL") or _DEFAULT_FILTER


def summarize_model() -> str:
    return os.environ.get("PF_SUMMARIZE_MODEL") or _DEFAULT_SUMMARIZE


def reset_client() -> None:
    """Drop the cached boto3 client (e.g. after auth failure / key rotation)."""
    global _client
    _client = None


@contextmanager
def _bearer_preferred_env():
    """While creating a bearer-auth client, hide profile/SSO that pods lack.

    Hosted Training often injects AWS_PROFILE from a local env_file. Leaving it
    set after client creation can make later Converse calls prefer a missing
    profile over AWS_BEARER_TOKEN_BEDROCK. Keep profile suppressed for the
    process lifetime when a bearer key is configured.
    """
    saved: dict[str, str] = {}
    if _bedrock_api_key():
        for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
            if name in os.environ:
                saved[name] = os.environ.pop(name)
    try:
        yield
    finally:
        # Do NOT restore profiles when bearer is set — that re-breaks auth.
        if not _bedrock_api_key():
            os.environ.update(saved)


def _get_client():
    global _client
    if _client is not None:
        return _client
    import boto3
    from botocore.config import Config

    _ensure_bearer_env()
    with _bearer_preferred_env():
        config = Config(
            read_timeout=90,
            connect_timeout=10,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        _client = boto3.client("bedrock-runtime", region_name=bedrock_region(), config=config)
    return _client


def _extract_converse_text(resp: dict) -> str:
    blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("text"):
            parts.append(str(b["text"]).strip())
    return "\n".join(parts)


def _is_auth_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "AccessDenied" in name or "UnrecognizedClient" in name or "ExpiredToken" in name:
        return True
    return any(
        needle in msg
        for needle in (
            "accessdenied",
            "api key is valid",
            "security token",
            "expired",
            "unauthorized",
            "invalidsecurity",
            "could not refresh",
        )
    )


def _chat_sync(prompt: str, *, model: str, max_tokens: int) -> tuple[str, int, int]:
    client = _get_client()
    try:
        resp = client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            reset_client()
            raise BedrockAuthError(str(e)) from e
        raise
    text = _extract_converse_text(resp)
    usage = resp.get("usage", {})
    return text, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))


async def chat(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    role: str = "default",
) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens). ("", 0, 0) when disabled."""
    if not llm_enabled():
        return "", 0, 0
    if model is None:
        model = filter_model() if role == "filter" else summarize_model()
    try:
        return await asyncio.to_thread(_chat_sync, prompt, model=model, max_tokens=max_tokens)
    except BedrockAuthError:
        if _bedrock_api_key() and _has_iam_creds():
            # Expired/invalid bearer shadowing valid IAM keys: drop it once
            # for the process lifetime and retry on the IAM chain.
            _log.warning("Bedrock bearer token rejected; falling back to IAM credentials")
            _drop_bearer_env()
            reset_client()
            return await asyncio.to_thread(
                _chat_sync, prompt, model=model, max_tokens=max_tokens)
        raise
    except Exception as e:  # noqa: BLE001
        if _is_auth_error(e):
            reset_client()
            raise BedrockAuthError(str(e)) from e
        raise
