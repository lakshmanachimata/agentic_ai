"""
LangChain agent: image generation via Ollama ``x/z-image-turbo:latest``.

Uses Ollama ``POST /api/generate`` for image-capable models. Saves PNG files under
``generated_images/`` and writes ``latest_image.json`` for the Streamlit UI
(show + download).

Requires:
  ollama pull x/z-image-turbo:latest
  ollama pull qwen3.5:latest   # chat brain for this agent

Interactive:
  python image_agent.py

One-off:
  python image_agent.py "Generate an image of a rainy street in Tokyo at night"
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, make_chat_ollama, run_interactive

IMAGES_DIR = Path(__file__).resolve().parent / "generated_images"
LATEST_IMAGE_PATH = IMAGES_DIR / "latest_image.json"

DEFAULT_IMAGE_MODEL = "x/z-image-turbo:latest"
OLLAMA_GENERATE_URL = os.environ.get(
    "OLLAMA_GENERATE_URL",
    "http://127.0.0.1:11434/api/generate",
)
IMAGE_TIMEOUT_S = float(os.environ.get("IMAGE_GEN_TIMEOUT_S", "300"))


def _safe_filename(prompt: str) -> str:
    base = re.sub(r"[^\w\-.]+", "_", (prompt or "image").strip())[:50].strip("_")
    return base or "image"


def _extract_image_b64(payload: dict[str, Any]) -> str | None:
    """Pull base64 image bytes from various Ollama response shapes."""
    for key in ("image", "images"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                for k in ("image", "b64_json", "data"):
                    if isinstance(first.get(k), str) and first[k].strip():
                        return first[k].strip()
    # OpenAI-style nested data
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for k in ("b64_json", "image", "data"):
            if isinstance(data[0].get(k), str) and data[0][k].strip():
                return data[0][k].strip()
    return None


def _decode_image_bytes(b64_data: str) -> bytes:
    raw = b64_data.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _write_latest_meta(
    *,
    prompt: str,
    path: Path,
    model: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    meta = {
        "prompt": prompt,
        "path": str(path.resolve()),
        "filename": path.name,
        "model": model,
        "width": width,
        "height": height,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_IMAGE_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass


@tool
def generate_image(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 0,
    height: int = 0,
    seed: int = -1,
) -> str:
    """Generate an image from a text prompt using a local Ollama image model.

    Use when the user asks to create, draw, generate, or illustrate a picture.
    Write a clear, detailed English visual prompt (subject, setting, lighting, style).

    Args:
        prompt: Detailed image description (required).
        model: Ollama image model tag (default x/z-image-turbo:latest).
        width: Optional width in pixels (0 = model default).
        height: Optional height in pixels (0 = model default).
        seed: Optional seed (>=0) for reproducibility; -1 = random.
    """
    text = (prompt or "").strip()
    if not text:
        return "Error: prompt is required."

    model_name = (model or DEFAULT_IMAGE_MODEL).strip() or DEFAULT_IMAGE_MODEL
    body: dict[str, Any] = {
        "model": model_name,
        "prompt": text,
        "stream": False,
    }
    options: dict[str, Any] = {}
    if width and width > 0:
        options["width"] = int(width)
        body["width"] = int(width)
    if height and height > 0:
        options["height"] = int(height)
        body["height"] = int(height)
    if seed is not None and int(seed) >= 0:
        options["seed"] = int(seed)
        body["seed"] = int(seed)
    if options:
        body["options"] = options

    try:
        resp = httpx.post(OLLAMA_GENERATE_URL, json=body, timeout=IMAGE_TIMEOUT_S)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.TimeoutException:
        return (
            f"Error: image generation timed out after {IMAGE_TIMEOUT_S:.0f}s. "
            "Try a simpler prompt or raise IMAGE_GEN_TIMEOUT_S."
        )
    except httpx.HTTPError as e:
        return f"Error: Ollama image HTTP failure: {e}"
    except json.JSONDecodeError:
        return "Error: Ollama returned non-JSON for image generation."

    if isinstance(payload, dict) and payload.get("error"):
        return f"Error from Ollama: {payload['error']}"

    b64 = _extract_image_b64(payload if isinstance(payload, dict) else {})
    if not b64:
        keys = sorted((payload or {}).keys()) if isinstance(payload, dict) else []
        return (
            "Error: no image data in Ollama response. "
            f"Keys seen: {keys}. Ensure the model has image capability "
            f"(e.g. `ollama pull {DEFAULT_IMAGE_MODEL}`)."
        )

    try:
        img_bytes = _decode_image_bytes(b64)
    except Exception as e:
        return f"Error: could not decode image base64: {e}"

    if not img_bytes:
        return "Error: decoded image was empty."

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = IMAGES_DIR / f"{stamp}_{_safe_filename(text)}_{uuid.uuid4().hex[:6]}.png"
    out_path.write_bytes(img_bytes)

    _write_latest_meta(
        prompt=text,
        path=out_path,
        model=model_name,
        width=int(width) if width else None,
        height=int(height) if height else None,
    )

    lines = [
        "Image generated successfully.",
        f"Model: {model_name}",
        f"Prompt: {text}",
        f"Saved PNG: {out_path.resolve()}",
        f"Bytes: {len(img_bytes)}",
        "Open the Streamlit UI to preview and download the image.",
    ]
    return "\n".join(lines)


@tool
def list_generated_images() -> str:
    """List recently generated PNG images in the generated_images folder."""
    if not IMAGES_DIR.is_dir():
        return "No generated_images folder yet — create an image first."
    files = sorted(IMAGES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "generated_images/ exists but has no PNG files yet."
    lines = [f"Generated images ({len(files)}):"]
    for path in files[:20]:
        lines.append(f"- {path}")
    return "\n".join(lines)


def build_agent(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
):
    """Chat LLM plans the visual prompt; image model is always the tool default."""
    llm = make_chat_ollama(model=model, temperature=temperature, top_k=top_k)
    return create_agent(
        llm,
        tools=[generate_image, list_generated_images],
        system_prompt=(
            "You are an image-generation assistant backed by a local Ollama image model "
            f"({DEFAULT_IMAGE_MODEL}).\n"
            "- When the user wants a picture / illustration / poster / photo, call "
            "generate_image with a rich English prompt (subject, composition, lighting, style).\n"
            "- Improve vague requests into a concrete visual prompt, but keep the user's intent.\n"
            "- Do not invent that an image was created without calling the tool.\n"
            "- After success, report the saved PNG path clearly.\n"
            "- For listing prior files, call list_generated_images.\n"
            "Leave model empty unless the user names a different Ollama image tag."
        ),
        checkpointer=MemorySaver(),
    )


def run_query(graph: Any, question: str, *, thread_id: str | None = None) -> str:
    return invoke_agent(graph, question, thread_id=thread_id)


def main() -> None:
    graph = build_agent()
    q_one = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if q_one:
        print(run_query(graph, q_one))
        return

    run_interactive(
        "Image generation agent",
        f"describe a picture to generate with {DEFAULT_IMAGE_MODEL}.",
        graph,
    )


if __name__ == "__main__":
    main()
