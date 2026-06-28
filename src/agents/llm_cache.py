"""
Disk cache for LLM calls.

Every call is keyed by a hash of (model, system prompt, user prompt, options).
Because we run the model at temperature 0 (deterministic), a cached answer is
identical to a fresh one — so caching is free fidelity-wise and turns
re-running a backtest from hours into seconds. Cached results also make the
whole project reproducible by anyone who has the cache, even without a GPU.

The cache lives in cache/ (git-ignored by default). To SHARE results for
reproducibility we can later opt specific caches into the repo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ollama

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "llm"


def _key(model: str, system: str, user: str, options: dict) -> str:
    """Stable hash of everything that determines the model's output."""
    blob = json.dumps(
        {"model": model, "system": system, "user": user, "options": options},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def cached_chat(
    model: str,
    system: str,
    user: str,
    options: dict | None = None,
    use_json: bool = True,
) -> str:
    """Call the LLM through the cache. Returns the message content string.

    On a cache hit, reads instantly from disk. On a miss, calls the model,
    saves the result, and returns it.
    """
    options = options or {"temperature": 0.0}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = _key(model, system, user, options)
    path = CACHE_DIR / f"{key}.json"

    if path.exists():
        return json.loads(path.read_text())["content"]

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    if use_json:
        kwargs["format"] = "json"

    response = ollama.chat(**kwargs)
    content = response["message"]["content"]

    # Save the result alongside its inputs (for debugging/inspection)
    path.write_text(json.dumps(
        {"model": model, "system": system, "user": user,
         "options": options, "content": content},
        indent=2,
    ))
    return content


def cache_stats() -> dict:
    """How many calls are cached, and total size on disk."""
    if not CACHE_DIR.exists():
        return {"cached_calls": 0, "size_mb": 0.0}
    files = list(CACHE_DIR.glob("*.json"))
    size_mb = sum(f.stat().st_size for f in files) / 1_048_576
    return {"cached_calls": len(files), "size_mb": round(size_mb, 3)}
