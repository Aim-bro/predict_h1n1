import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# 디스크 캐시 위치
CACHE_DIR = Path("data/cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def gemini_generate(prompt: str, model: str = "gemini-1.5-pro", use_cache: bool = True) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다")

    cache_key = _hash(model + "||" + prompt)
    cache_path = CACHE_DIR / f"{cache_key}.json"

    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    text = resp.text or ""

    if use_cache:
        cache_path.write_text(json.dumps({"model": model, "text": text}, ensure_ascii=False, indent=2), encoding="utf-8")

    return text
