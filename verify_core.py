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


class DocumentVerifier:
    """자료로 NM을 학습하고, LLM 답변을 문장 단위로 검증."""

    def __init__(self):
        self.nm = None
        self.trained = False
        self.corpus_preview = ""

    def train_on_document(self, corpus_text, embedding_dim=32):
        """LLM에 준 것과 '같은 자료'로 NM 학습."""
        self.nm = NeuralMarkovEngine()
        self.nm.train(corpus_text, embedding_dim=embedding_dim)
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

            # 판정=설명: 상태에 따라 색과 근거를 함께
            if status == "PASS":
                color, label = "green", "근거 있음"
            elif status == "WARNING":
                color, label = "orange", "경계(일부 이탈)"
            elif status in ("FATAL", "CRITICAL"):
                color, label = "red", "자료에서 벗어남"
            else:  # SKIP (너무 짧음)
                color, label = "gray", "판정 보류(짧음)"

            # 설명: 어느 토큰이 가장 근거가 약한지(가장 낮은 logP 토큰)
            weak = ""
            per = ev.get("per_token", [])
            if per:
                lowest = min(per, key=lambda p: p.get("logp", 0))
                if lowest.get("logp", 0) < -10:
                    weak = lowest.get("raw_token", lowest.get("token", ""))

            note = f"{label} (logP {logp:.1f})"
            if weak:
                note += f" · 근거 약한 표현: '{weak}'"

            results.append({
                "sentence": s, "status": status, "avg_logp": round(logp, 2),
                "color": color, "label": label, "note": note,
                "weak_token": weak,
            })
        return results

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
