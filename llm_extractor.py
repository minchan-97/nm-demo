"""
llm_extractor.py — LLM으로 답변에서 항목-값 claim 추출.

블랙박스를 '가두는' 설계:
  - LLM은 후보(claim)만 생성한다. 판정하지 않는다.
  - 각 claim에 '답변 원문의 근거 문장'을 강제로 인용하게 한다.
  - 추출 결과는 반드시 답변 텍스트에 실재하는지 재확인(할루 방지).
  → LLM이 지어낸 claim은 재확인에서 걸러진다.

LLM은 '읽어서 구조화'만(scribe). 정오 판정은 claim_verify가 자료로 한다.
"""
from __future__ import annotations
import json
import re
from claim_verify import Claim, detect_value_type


_EXTRACT_PROMPT = """다음은 어떤 문서에 대한 답변입니다. 이 답변에서 '항목-값' 형태의
사실 주장만 뽑아주세요. 특히 날짜, 숫자, 인원, 명칭 같은 검증 가능한 값에 집중하세요.

규칙:
- 각 주장은 반드시 답변에 실제로 있는 문장에서만 뽑으세요. 지어내지 마세요.
- 항목(item)은 무엇에 대한 것인지, 값(value)은 그 항목의 구체적 값입니다.
- 근거(quote)는 그 주장이 나온 답변 원문 문장을 그대로 복사하세요.
- JSON 배열로만 답하세요. 다른 말/마크다운/설명 없이.

형식:
[{{"item": "항목", "value": "값", "quote": "답변 원문 문장"}}]

답변:
\"\"\"
{answer}
\"\"\"

JSON:"""


def make_openai_extractor(api_key: str, model: str = "gpt-4o-mini"):
    """
    OpenAI 기반 claim 추출기를 반환.
    반환된 함수는 answer_text -> [Claim] 형태.
    """
    def extractor(answer_text: str):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": _EXTRACT_PROMPT.format(answer=answer_text[:4000])}],
                temperature=0.0,  # 추출은 결정적으로
            )
            raw = resp.choices[0].message.content
            return _parse_claims(raw, answer_text)
        except Exception as e:
            # 실패 시 빈 리스트(폴백은 호출측에서 규칙 추출로)
            print(f"[llm_extractor] 실패: {e}")
            return []
    return extractor


def _parse_claims(raw: str, answer_text: str):
    """LLM 응답(JSON)을 Claim으로. 근거가 답변에 실재하는지 확인."""
    # 코드블록/잡텍스트 제거
    s = raw.strip()
    s = re.sub(r'^```(json)?', '', s).strip()
    s = re.sub(r'```$', '', s).strip()
    m = re.search(r'\[.*\]', s, re.DOTALL)
    if m:
        s = m.group(0)
    try:
        data = json.loads(s)
    except Exception:
        return []

    answer_compact = re.sub(r'\s+', '', answer_text)
    claims = []
    for d in data:
        if not isinstance(d, dict):
            continue
        item = str(d.get("item", "")).strip()
        value = str(d.get("value", "")).strip()
        quote = str(d.get("quote", "")).strip()
        if not item or not value:
            continue
        # 블랙박스 가두기: 근거(quote)나 값이 답변에 실재하는지 확인
        # (LLM이 지어낸 claim 걸러내기)
        quote_compact = re.sub(r'\s+', '', quote)
        value_compact = re.sub(r'\s+', '', value)
        if quote_compact and quote_compact not in answer_compact:
            # 근거가 답변에 없음 → 지어낸 것으로 보고 스킵
            if value_compact not in answer_compact:
                continue
        vtype = detect_value_type(value)
        # claim의 sentence는 quote 우선, 없으면 값이 든 문장
        sentence = quote if quote else _find_sentence(answer_text, value)
        claims.append(Claim(item=item, value=value, value_type=vtype,
                            sentence=sentence, evidence_quote=quote))
    return claims


def _find_sentence(text: str, value: str) -> str:
    for sent in re.split(r'(?<=[.!?。])\s+|\n+', text):
        if value in sent:
            return sent.strip()
    return value
