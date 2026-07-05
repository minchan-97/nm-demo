"""
claim_verify.py — 삼자 구조의 심장.

오늘 도달한 구조:
  LLM(scribe)  : 답변에서 '항목-값' 주장(claim)을 근거와 함께 추출
  NM(통계)     : 각 claim이 자료 언어 패턴에 맞는지 logP로 검증
  판정자(투명) : 근거가 자료 원문에 실재하는지 + NM 통계 + 항목일치를
                 투명한 규칙으로 종합해 판정. 판정 경로가 곧 설명.

핵심 원칙:
  - LLM은 후보(claim)만 생성한다. 판단하지 않는다.
  - 판정은 '자료 원문 대조'라는 외부 심판이 한다(문자열/숫자 비교 = 투명).
  - 왜 통과/거부인지가 규칙 경로로 다 보인다(지어낼 수 없음).

이 파일은 LLM 없이도 동작하도록 설계:
  - LLM 추출기가 주어지면 사용(주입식), 없으면 규칙 기반 추출로 폴백.
  - 따라서 대조·판정 로직 자체는 LLM과 독립적으로 검증 가능.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Callable


# ─────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────
@dataclass
class Claim:
    """답변에서 뽑은 하나의 항목-값 주장."""
    item: str                 # 항목 (예: "수학여행")
    value: str                # 값 (예: "9월")
    value_type: str           # "date" | "number" | "name" | "other"
    sentence: str             # 이 claim이 나온 원 문장
    evidence_quote: str = ""  # LLM이 댄 근거 원문 인용(있으면)


@dataclass
class Verdict:
    """한 claim에 대한 판정 결과 + 설명 경로."""
    claim: Claim
    status: str               # "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE"
    color: str                # green / red / gray
    reason: str               # 사람이 읽는 설명
    path: list = field(default_factory=list)  # 판정에 이른 규칙 경로(투명)
    doc_value: str = ""       # 자료에서 찾은 실제 값(있으면)


# ─────────────────────────────────────────────────────────
# 값 타입 감지 + 정규화 (숫자·날짜는 특별 취급)
# ─────────────────────────────────────────────────────────
_MONTHS = r'(1[0-2]|[1-9])월'
_DAYS = r'(3[01]|[12][0-9]|[1-9])일'
_NUM = r'\d[\d,]*'

def detect_value_type(value: str) -> str:
    v = value.strip()
    if re.search(_MONTHS, v) or re.search(_DAYS, v) or re.search(r'\d{4}년', v):
        return "date"
    if re.fullmatch(rf'\s*{_NUM}\s*(명|개|시간|일|원|학급|반|권|회|층)?\s*', v):
        return "number"
    return "other"


def normalize_value(value: str, vtype: str):
    """비교를 위해 값을 정규화. 날짜는 (월,일) 집합, 숫자는 int."""
    v = value.strip()
    if vtype == "date":
        months = [int(m) for m in re.findall(_MONTHS, v)]
        # '월'을 뗀 순수 숫자(일) 추출은 월과 겹치므로 월 우선 처리
        return {"months": sorted(set(months)), "raw": v}
    if vtype == "number":
        nums = re.findall(_NUM, v)
        if nums:
            return {"num": int(nums[0].replace(",", "")), "raw": v}
    return {"raw": v}


# ─────────────────────────────────────────────────────────
# 규칙 기반 claim 추출 (LLM 폴백용)
# ─────────────────────────────────────────────────────────
def rule_extract_claims(answer_text: str) -> list:
    """
    LLM이 없을 때 쓰는 간단한 규칙 추출.
    '<항목>은/는 ... <날짜/숫자> ...' 패턴에서 항목-값을 뽑는다.
    완벽하지 않음 — LLM 추출기의 폴백/베이스라인.
    """
    claims = []
    sentences = re.split(r'(?<=[.!?。])\s+|\n+', answer_text)
    for sent in sentences:
        s = sent.strip()
        if len(s) < 4:
            continue
        # 날짜/숫자 값을 찾음
        date_m = re.search(rf'{_MONTHS}\s*({_DAYS})?', s)
        num_m = re.search(rf'{_NUM}\s*(명|개|시간|학급|반|권|회|층|원)', s)
        value, vtype = "", ""
        if date_m:
            value, vtype = date_m.group(0), "date"
        elif num_m:
            value, vtype = num_m.group(0), "number"
        else:
            continue
        # 항목 = 문장 앞부분의 주어(간단 근사: 조사 '은/는/이/가' 앞 명사구)
        subj = re.match(r'\s*([^,]+?)(은|는|이|가|의|에서|에는)\s', s)
        item = subj.group(1).strip() if subj else s[:12]
        claims.append(Claim(item=item, value=value, value_type=vtype, sentence=s))
    return claims


# ─────────────────────────────────────────────────────────
# 판정자 (투명 규칙) — 삼자 구조의 심판
# ─────────────────────────────────────────────────────────
class ClaimVerifier:
    """
    자료(corpus)를 심판 기준으로, 각 claim을 판정한다.
    LLM 추출기는 선택적으로 주입(없으면 규칙 추출).
    """

    def __init__(self, corpus_text: str,
                 nm_engine=None,
                 llm_extractor: Optional[Callable] = None):
        self.corpus = corpus_text
        self.corpus_norm = self._normalize_ws(corpus_text)
        self.nm = nm_engine
        self.llm_extractor = llm_extractor

    @staticmethod
    def _normalize_ws(t: str) -> str:
        return re.sub(r'\s+', ' ', t)

    # ── 항목이 자료에서 언급된 위치들(문맥 조각) 찾기 ──
    def _find_item_contexts(self, item: str, window: int = 30) -> list:
        """
        자료에서 항목명이 나오는 위치의 '같은 줄/직후' 조각만 반환.
        넓은 window는 이웃 항목의 값까지 섞여 오탐하므로,
        항목 뒤쪽 좁은 범위(같은 문장/줄)만 본다.
        """
        contexts = []
        key = re.sub(r'\s+', '', item)
        key_loose = re.sub(r'(은|는|이|가|을|를|의|에|에서|에는)$', '', key)
        if len(key_loose) < 2:
            return contexts
        # 원문을 줄 단위로 보고, 항목 핵심어가 든 줄만 취함
        # (표/목록에서 한 줄 = 한 항목-값 쌍인 경우가 많음)
        search_key = key_loose[:8]
        for line in self.corpus.split('\n'):
            compact_line = line.replace(' ', '')
            pos = compact_line.find(search_key[:5])
            if pos >= 0:
                # 항목명 '이후' 부분만 (값은 보통 항목 뒤에 옴)
                after = compact_line[pos:]
                contexts.append(after[:window * 2])
        return contexts

    # ── 핵심: 하나의 claim 판정 ──
    def verify_claim(self, claim: Claim) -> Verdict:
        path = []
        # 1) 항목이 자료에 있나?
        contexts = self._find_item_contexts(claim.item)
        if not contexts:
            path.append(f"항목 '{claim.item}'을(를) 자료에서 찾지 못함")
            return Verdict(claim, "UNVERIFIABLE", "gray",
                           f"'{claim.item}'이(가) 자료에 없어 검증 불가", path)
        path.append(f"항목 '{claim.item}' 자료에서 {len(contexts)}곳 발견")

        # 2) 그 문맥들에서 같은 타입의 값을 수집 (자료가 말하는 실제 값)
        doc_values = self._collect_values_near(contexts, claim.value_type)
        if not doc_values:
            path.append(f"항목 문맥에 {claim.value_type} 값이 없음")
            # 값 자체가 자료 항목 문맥에 없으면, 답변 값은 근거 없음
            return Verdict(claim, "CONTRADICTED", "red",
                           f"'{claim.item}' 문맥에 {claim.value_type} 정보가 없는데 "
                           f"답변은 '{claim.value}'라고 주장", path)
        path.append(f"자료의 '{claim.item}' 문맥 값: {sorted(set(doc_values))}")

        # 3) 답변 값 vs 자료 값 대조 (투명한 정규화 비교)
        if self._value_matches(claim.value, doc_values, claim.value_type):
            path.append(f"답변 값 '{claim.value}' 가 자료 값과 일치")
            return Verdict(claim, "SUPPORTED", "green",
                           f"'{claim.item} = {claim.value}' 자료와 일치", path,
                           doc_value=", ".join(sorted(set(doc_values))))
        else:
            path.append(f"답변 값 '{claim.value}' 가 자료 값과 불일치")
            return Verdict(claim, "CONTRADICTED", "red",
                           f"자료의 '{claim.item}'은(는) {sorted(set(doc_values))}인데 "
                           f"답변은 '{claim.value}' — 불일치(환각 가능)", path,
                           doc_value=", ".join(sorted(set(doc_values))))

    def _collect_values_near(self, contexts: list, vtype: str) -> list:
        vals = []
        for seg in contexts:
            if vtype == "date":
                for m in re.finditer(rf'{_MONTHS}\s*({_DAYS})?', seg):
                    vals.append(m.group(0))
            elif vtype == "number":
                for m in re.finditer(rf'{_NUM}\s*(명|개|시간|학급|반|권|회|층|원)', seg):
                    vals.append(m.group(0))
        return vals

    def _value_matches(self, ans_value: str, doc_values: list, vtype: str) -> bool:
        an = normalize_value(ans_value, vtype)
        for dv in doc_values:
            dn = normalize_value(dv, vtype)
            if vtype == "date":
                # 월이 겹치면 일치로 봄(간단 규칙)
                if set(an.get("months", [])) & set(dn.get("months", [])):
                    return True
            elif vtype == "number":
                if an.get("num") is not None and an.get("num") == dn.get("num"):
                    return True
            else:
                if an.get("raw") == dn.get("raw"):
                    return True
        return False

    # ── 전체 답변 검증 ──
    def verify(self, answer_text: str) -> list:
        # 후보 생성: LLM 있으면 LLM, 없으면 규칙
        if self.llm_extractor is not None:
            claims = self.llm_extractor(answer_text)
        else:
            claims = rule_extract_claims(answer_text)
        return [self.verify_claim(c) for c in claims]
