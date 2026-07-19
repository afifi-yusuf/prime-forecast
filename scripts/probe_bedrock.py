#!/usr/bin/env python3
"""Probe Bedrock leak-filter credentials (does not print the key).

Usage:
  set -a && source secrets.env && set +a
  python scripts/probe_bedrock.py
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "prime_forecast"))


def _inspect_key(key: str) -> None:
    print(f"key_present: yes (len={len(key)})")
    if not key.startswith("bedrock-api-key-"):
        print("key_kind: opaque/long-term (no bedrock-api-key- prefix metadata)")
        return
    b64 = key[len("bedrock-api-key-"):]
    pad = "=" * (-len(b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(b64 + pad).decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"key_kind: bedrock-api-key (decode failed: {e})")
        return
    m_date = re.search(r"X-Amz-Date=(\d{8}T\d{6}Z)", decoded)
    m_exp = re.search(r"X-Amz-Expires=(\d+)", decoded)
    if not (m_date and m_exp):
        print("key_kind: bedrock-api-key (no expiry fields found)")
        return
    start = datetime.strptime(m_date.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    exp = start + timedelta(seconds=int(m_exp.group(1)))
    now = datetime.now(timezone.utc)
    print("key_kind: SHORT-TERM bearer (has X-Amz-Expires)")
    print(f"issued_utc:  {start.isoformat()}")
    print(f"expires_utc: {exp.isoformat()}")
    print(f"ttl_hours:   {int(m_exp.group(1)) / 3600:g}")
    print(f"expired:     {now > exp}")
    if now > exp:
        print("ACTION: mint a LONG-TERM Bedrock API key and update secrets + Hub.")


async def _main() -> int:
    from prime_forecast.bedrock import (  # noqa: WPS433
        BedrockAuthError,
        _bedrock_api_key,
        chat,
        filter_model,
        llm_enabled,
        reset_client,
    )

    key = _bedrock_api_key()
    if not key:
        print("key_present: no")
        print("Set AWS_BEARER_TOKEN_BEDROCK or BEDROCK_API_KEY in secrets.env")
        return 1
    _inspect_key(key)
    print(f"llm_enabled: {llm_enabled()}")
    print(f"region:      {os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')}")
    print(f"model:       {filter_model()}")
    if os.environ.get("AWS_PROFILE"):
        print(f"WARNING: AWS_PROFILE={os.environ['AWS_PROFILE']!r} is set; "
              "Hosted Training pods usually lack SSO profiles.")

    reset_client()
    try:
        text, inp, out = await chat("Reply with exactly: OK", max_tokens=16, role="filter")
    except BedrockAuthError as e:
        print(f"converse: AUTH_FAIL ({e})")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"converse: ERROR ({type(e).__name__}: {e})")
        return 3

    print(f"converse: OK  in={inp} out={out} reply={text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
