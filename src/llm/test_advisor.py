import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors


# 디스크 캐시(같은 프롬프트 재호출 시 API 호출 스킵)
CACHE_DIR = Path("data/cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_prompt(project_context: Dict[str, Any], latest_result: Dict[str, Any]) -> str:
    return f"""
너는 시니어 데이터 분석가이자 ML 실험 매니저다.
목표는 F1을 최대화하는 것이다. 단, 학습 루프 내부 호출은 금지이며 "다음 실험 계획"만 제안하라.

[프로젝트 컨텍스트]
{json.dumps(project_context, ensure_ascii=False, indent=2)}

[최근 실험 결과]
{json.dumps(latest_result, ensure_ascii=False, indent=2)}

요구사항:
1) 다음 실험 Top-5를 우선순위로 제안
2) 각 실험에 대해 (기대효과, 리스크, 실행시간) 추정치 포함 (추정은 '(추정)'이라고 명시)
3) 지금 코드 구조(CatBoost/LGBM, threshold 최적화)를 유지하는 선에서 제안
4) 출력은 JSON 하나로만

출력 JSON 스키마:
{{
  "next_steps": [
    {{
      "title": "...",
      "why": "...",
      "how": "...",
      "expected_gain_f1": "...(추정)",
      "time_cost": "...(추정)",
      "risk": "..."
    }}
  ]
}}
""".strip()


def run_llm_advisor(
    project_context: Dict[str, Any],
    latest_result: Dict[str, Any],
    model: str = "gemini-2.0-flash",
    use_cache: bool = True,
) -> str:
    import time
    import json
    import os
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types, errors

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다 (프로젝트 루트에 .env 확인)")

    prompt = build_prompt(project_context, latest_result)

    # ---------- 캐시 ----------
    cache_key = _hash(model + "||" + prompt)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

    # ---------- Gemini Client ----------
    client = genai.Client(api_key=api_key)

    # ---------- 재시도 로직 ----------
    max_retries = 6
    sleep_s = 2.0
    text = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                ),
            )
            text = resp.text or ""
            break

        except errors.ServerError as e:
            # 503 UNAVAILABLE (과부하)
            if attempt == max_retries:
                raise
            time.sleep(sleep_s)
            sleep_s *= 1.8

    # ---------- 캐시 저장 ----------
    if use_cache and text:
        cache_path.write_text(
            json.dumps({"model": model, "text": text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return text



if __name__ == "__main__":
    # main.py 결과(res)를 흉내낸 더미. 연결 테스트 용도
    dummy_project_context = {
        "metric": "F1",
        "cv": "StratifiedKFold 5",
        "note": "LLM은 학습 루프 밖에서만 사용",
    }

    dummy_latest_result = {
        "target": "vacc_h1n1_f",
        "mean_f1": 0.63497,
        "std_f1": 0.00425,
        "best_model": "CatBoost depth=7",
        "runtime_minutes": 137.61,
    }

    advice = run_llm_advisor(dummy_project_context, dummy_latest_result, use_cache=True)
    print("\n[Gemini Advisor Output]\n")
    print(advice)
