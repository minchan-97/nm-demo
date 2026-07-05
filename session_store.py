"""
session_store.py — 검증 세션을 pkl로 저장/로드.

목적: 피드백 및 문제점 파악.
  실제 문서로 돌려본 결과(자료·답변·NM판정·claim판정)를 남겨두고,
  나중에 "어디서 틀렸나"를 되짚는다. 특히:
    - NM이 놓친 것 / claim이 놓친 것
    - 오탐(정상을 환각으로) / 미탐(환각을 통과)
    - 사람이 단 피드백(맞음/틀림 라벨)

저장 구조는 사람이 나중에 분석하기 쉽게 평평한 dict 리스트.
"""
from __future__ import annotations
import pickle
import os
import time
from dataclasses import dataclass, field, asdict


@dataclass
class VerificationSession:
    """한 번의 검증 세션 전체 기록."""
    corpus_preview: str                    # 자료 앞부분(전체는 크니 미리보기)
    answer_text: str                       # 검증한 답변
    nm_results: list = field(default_factory=list)      # 층1 결과
    claim_results: list = field(default_factory=list)   # 층2 결과
    feedback: dict = field(default_factory=dict)         # 사람 피드백
    meta: dict = field(default_factory=dict)             # 시각, 문서명 등
    created_at: float = field(default_factory=time.time)

    def add_feedback(self, target_id: str, correct: bool, note: str = ""):
        """
        판정에 대한 사람 피드백.
        target_id: 'nm:<i>' 또는 'claim:<i>' (몇 번째 판정인가)
        correct: 그 판정이 맞았나
        note: 자유 메모
        """
        self.feedback[target_id] = {"correct": correct, "note": note,
                                    "at": time.time()}


def save_session(session: VerificationSession, path: str):
    """세션을 pkl로 저장."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(asdict(session), f)
    return path


def load_session(path: str) -> dict:
    """pkl에서 세션 로드(dict 형태)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def list_sessions(folder: str) -> list:
    """폴더의 세션 pkl 목록(경로, 시각, 답변 미리보기)."""
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".pkl"):
            continue
        p = os.path.join(folder, name)
        try:
            d = load_session(p)
            out.append({
                "path": p,
                "name": name,
                "created_at": d.get("created_at"),
                "answer_preview": (d.get("answer_text", "") or "")[:50],
                "n_nm": len(d.get("nm_results", [])),
                "n_claim": len(d.get("claim_results", [])),
                "n_feedback": len(d.get("feedback", {})),
            })
        except Exception:
            continue
    return out


def analyze_problems(session: dict) -> dict:
    """
    저장된 세션에서 문제점 자동 요약.
    피드백이 있으면 오탐/미탐을 집계.
    """
    fb = session.get("feedback", {})
    problems = {"오탐": [], "미탐": [], "확인안됨": 0}
    nm = session.get("nm_results", [])
    claim = session.get("claim_results", [])

    def _judge_is_flag(color):
        # 빨강/주황 = '문제 있다'고 판정한 것
        return color in ("red", "orange")

    for i, r in enumerate(nm):
        key = f"nm:{i}"
        if key in fb:
            correct = fb[key]["correct"]
            flagged = _judge_is_flag(r.get("color", ""))
            if not correct and flagged:
                problems["오탐"].append(("nm", i, r.get("sentence", "")[:40]))
            elif not correct and not flagged:
                problems["미탐"].append(("nm", i, r.get("sentence", "")[:40]))
    for i, r in enumerate(claim):
        key = f"claim:{i}"
        if key in fb:
            correct = fb[key]["correct"]
            flagged = _judge_is_flag(r.get("color", ""))
            if not correct and flagged:
                problems["오탐"].append(("claim", i, f"{r.get('item')}={r.get('value')}"))
            elif not correct and not flagged:
                problems["미탐"].append(("claim", i, f"{r.get('item')}={r.get('value')}"))

    problems["확인안됨"] = (len(nm) + len(claim)) - len(fb)
    return problems
