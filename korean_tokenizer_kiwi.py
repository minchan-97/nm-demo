"""
korean_tokenizer_kiwi.py — Kiwi 형태소 분석 기반 토크나이저.

어미 문제 해결:
  기존 규칙 토크나이저는 "갑니다"를 통짜로 남겨 자료의 "간다"와 매칭 실패.
  Kiwi는 "가/VV + ㅂ니다/EF"로 분석 → 어간 "가"만 취해 매칭 성공.
  (불규칙 활용도 Kiwi가 처리)

핵심 설계:
  - 의미를 지닌 형태소(명사·동사·형용사·부사 어간)만 취한다.
  - 조사·어미·문장부호는 버린다(도메인 판정에 노이즈).
  - Kiwi 미설치 환경에선 기존 규칙 토크나이저로 자동 폴백.
"""
from __future__ import annotations

# 의미를 지닌 품사만 취함 (어간 중심)
#  NNG 일반명사, NNP 고유명사, NNB 의존명사, NR 수사, NP 대명사
#  VV 동사, VA 형용사, VX 보조용언, MAG 일반부사, MAJ 접속부사
#  XR 어근, SL 외국어, SN 숫자
_CONTENT_POS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "VV", "VA", "VX", "MAG", "MAJ", "XR", "SL", "SN", "SH",
}

_kiwi = None
_kiwi_tried = False


def _get_kiwi():
    """Kiwi 인스턴스 lazy 로드. 없으면 None."""
    global _kiwi, _kiwi_tried
    if _kiwi_tried:
        return _kiwi
    _kiwi_tried = True
    try:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    except Exception:
        _kiwi = None
    return _kiwi


def tokenize(text):
    """
    형태소 분석 후 내용 형태소의 '어간'만 반환.
    Kiwi 있으면 정확한 어간 추출, 없으면 규칙 토크나이저 폴백.
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        # 폴백: 기존 규칙 토크나이저
        from korean_tokenizer import tokenize as rule_tokenize
        return rule_tokenize(text)

    tokens = []
    for tok in kiwi.tokenize(text):
        if tok.tag in _CONTENT_POS:
            # 동사/형용사는 원형 어간(form), 숫자는 NUM 정규화
            if tok.tag == "SN":
                tokens.append("NUM")
            else:
                tokens.append(tok.form)
    return tokens


def tokenize_dual(text):
    """
    (원문토큰, 정규화어간) 쌍. NM 엔진이 raw_token 표시에 사용.
    Kiwi는 형태소 단위라, 원문 표층형과 어간을 함께 준다.
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        from korean_tokenizer import tokenize_dual as rule_dual
        return rule_dual(text)

    pairs = []
    for tok in kiwi.tokenize(text):
        if tok.tag in _CONTENT_POS:
            norm = "NUM" if tok.tag == "SN" else tok.form
            pairs.append((tok.form, norm))
    return pairs


def kiwi_available():
    """현재 Kiwi가 로드됐는지."""
    return _get_kiwi() is not None
