"""
verify_core.py — "판정=설명" 데모의 심장.

구조:
  같은 자료를 LLM과 NM에 모두 준다.
  LLM이 그 자료로 답변 생성 → NM이 답변의 각 '문장'을 판정.
  판정(색)과 설명(logP·근거)이 한 몸으로 나온다.

핵심: NM은 LLM과 독립적으로, '준 자료 안에 이 문장이 근거를 두는가'를 본다.
  LLM이 자료에 없는 걸 지어내면 → NM의 logP가 낮아짐 → 빨강.
  이게 SHAP/LLM자기검증과 다른 점: 판정 계산 자체가 곧 설명이라 지어낼 수 없음.
"""
from __future__ import annotations
import re
import sys, os
sys.path.append(os.path.dirname(__file__))
from neural_markov_engine import NeuralMarkovEngine
from claim_verify import ClaimVerifier


# ── 문장 분리 (한국어 대응) ─────────────────────────────────
def split_sentences(text):
    """
    한국어 문장 분리. 종결어미·문장부호 기준.
    완벽하진 않지만 데모에 충분하게 견고히.
    """
    text = text.strip()
    if not text:
        return []
    # 줄바꿈은 일단 공백화(단 목록은 유지)
    # 문장 끝 패턴: .!? 또는 한국어 종결(다./요./음.) 뒤 공백
    # 약어·소수점 오분리 방지: 숫자.숫자는 안 자름
    parts = re.split(r'(?<=[.!?。])\s+|(?<=[다요음])\.\s+|\n{2,}', text)
    sents = []
    for p in parts:
        p = p.strip()
        if len(p) >= 2:
            sents.append(p)
    return sents


import sys, os
sys.path.append(os.path.dirname(__file__))


class DocumentVerifier:
    """
    삼자 구조 검증기:
      1) NM(통계): 문장이 자료 언어 패턴에서 벗어났나 (자료에 없는 표현)
      2) claim(항목-값): 답변의 항목-값 주장이 자료와 정오 일치하나 (수학여행 9월 vs 10월)
      3) 판정자: 두 층을 투명 규칙으로 종합. 판정 경로가 곧 설명.
    """

    def __init__(self):
        self.nm = None
        self.trained = False
        self.corpus_preview = ""
        self.corpus_text = ""
        self.claim_verifier = None

    def train_on_document(self, corpus_text, embedding_dim=32, llm_extractor=None):
        """LLM에 준 것과 '같은 자료'로 NM 학습 + claim 심판 준비."""
        self.nm = NeuralMarkovEngine()
        self.nm.train(corpus_text, embedding_dim=embedding_dim)
        self.corpus_text = corpus_text
        self.claim_verifier = ClaimVerifier(
            corpus_text, nm_engine=self.nm, llm_extractor=llm_extractor)
        self.trained = True
        self.corpus_preview = corpus_text[:200]
        return True

    def verify_answer(self, answer_text):
        """
        LLM 답변 → 문장별 판정.
        반환: [{sentence, status, avg_logp, color, note, evidence}]
        """
        if not self.trained:
            raise RuntimeError("먼저 자료로 NM을 학습하세요.")
        sents = split_sentences(answer_text)
        results = []
        for s in sents:
            ev = self.nm.evaluate(s)
            status = ev["status"]
            logp = ev.get("avg_logp", 0.0)
            z = ev.get("z_score", None)

            # 판정=설명: 상태에 따라 색과 근거를 함께
            if status == "PASS":
                color, label = "green", "근거 있음"
            elif status == "WARNING":
                color, label = "orange", "경계(일부 이탈)"
            elif status in ("FATAL", "CRITICAL"):
                color, label = "red", "자료에서 벗어남"
            else:  # SKIP (너무 짧음)
                color, label = "gray", "판정 보류(짧음)"

            # 설명: 가장 근거 약한 토큰 + PMI 보정 이유(투명)
            weak = ""
            per = ev.get("per_token", [])
            if per:
                lowest = min(per, key=lambda p: p.get("logp", 0))
                if lowest.get("logp", 0) < -10:
                    weak = lowest.get("raw_token", lowest.get("token", ""))

            # 판정 근거: z-score(자료 분포 대비 이탈)를 명시 — 판정=설명
            if z is not None:
                note = f"{label} (자료 분포 대비 {abs(z):.1f}σ 이탈, logP {logp:.1f})"
            else:
                note = f"{label} (logP {logp:.1f})"
            if weak:
                note += f" · 근거 약한 표현: '{weak}'"

            results.append({
                "sentence": s, "status": status, "avg_logp": round(logp, 2),
                "z_score": z, "color": color, "label": label, "note": note,
                "weak_token": weak,
            })
        return results

    def verify_claims(self, answer_text):
        """
        항목-값 주장(claim) 검증. NM이 못 잡는 '자료에 있는 단어의
        틀린 조합'(수학여행 9월 vs 10월)을 자료 대조로 잡는다.
        반환: [{item, value, status, color, reason, path, doc_value}]
        """
        if not self.trained:
            raise RuntimeError("먼저 자료로 학습하세요.")
        verdicts = self.claim_verifier.verify(answer_text)
        out = []
        for v in verdicts:
            out.append({
                "item": v.claim.item,
                "value": v.claim.value,
                "value_type": v.claim.value_type,
                "sentence": v.claim.sentence,
                "status": v.status,
                "color": v.color,
                "reason": v.reason,
                "path": v.path,
                "doc_value": v.doc_value,
            })
        return out

    def summary(self, results):
        n = len(results)
        if n == 0:
            return {}
        by = {}
        for r in results:
            by[r["status"]] = by.get(r["status"], 0) + 1
        risky = sum(1 for r in results if r["color"] == "red")
        return {
            "총_문장": n,
            "근거있음": by.get("PASS", 0),
            "경계": by.get("WARNING", 0),
            "벗어남": by.get("FATAL", 0) + by.get("CRITICAL", 0),
            "위험_비율": round(risky / n, 2) if n else 0,
        }
