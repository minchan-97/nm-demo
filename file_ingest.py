"""
file_ingest.py — 업로드 파일을 코퍼스 텍스트로 바꾸는 도우미
============================================================
- 전문 코퍼스용: txt / pdf / docx 에서 본문 텍스트 추출 (LLM 불필요)
- 학생 그림용  : jpg/png/pdf 이미지를 GPT-4V로 '쉬운 말 설명'으로 변환 (LLM 필요)

원칙: 텍스트 추출은 CPU만으로. 이미지 해석만 LLM(GPT-4V) 사용.
      이미지 해석 결과는 선생님이 화면에서 확인·수정한 뒤 코퍼스에 넣는다.
"""
from __future__ import annotations
import io, base64


# ── 전문 코퍼스: 파일 → 텍스트 (LLM 불필요) ──────────────
def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    if name.endswith(".docx"):
        return _extract_docx(raw)
    # 알 수 없는 형식은 utf-8 시도
    return raw.decode("utf-8", errors="ignore")


def _looks_garbled(text: str) -> bool:
    """추출된 텍스트가 깨졌는지 감지(한글 PDF에서 폰트 인코딩 실패 시)."""
    if not text or len(text) < 20:
        return False
    # 정상 한글 음절(가-힣) + 영숫자 + 공백 비율
    import re
    total = len(text)
    valid = len(re.findall(r'[가-힣a-zA-Z0-9\s.,!?()\[\]:;~\-/월일년]', text))
    # 정상 문자 비율이 낮으면(깨진 글자·조합 자모 과다) 깨진 것으로 봄
    hangul_jamo = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', text))  # 조합 안된 자모(깨짐 신호)
    ratio = valid / total
    return ratio < 0.6 or hangul_jamo > total * 0.15


def _extract_pdf(raw: bytes) -> str:
    # pypdf 우선(pdfplumber보다 ~5배 빠름). 실패/깨짐 시 pdfplumber 폴백.
    pypdf_text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        pypdf_text = "\n".join(out)
        # 충분히 뽑혔고 + 깨지지 않았으면 사용
        if len(pypdf_text.strip()) >= 20 and not _looks_garbled(pypdf_text):
            return pypdf_text
    except Exception:
        pass
    # 폴백: pdfplumber (느리지만 한글 폰트에 더 견고한 경우 많음)
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                out.append(t)
        plumber_text = "\n".join(out)
        # 둘 중 안 깨진 쪽 선택
        if plumber_text.strip() and not _looks_garbled(plumber_text):
            return plumber_text
        # 둘 다 깨졌으면 → 경고 표시와 함께 덜 깨진 쪽 반환
        cand = max([pypdf_text, plumber_text], key=lambda t: len(t.strip()))
        if _looks_garbled(cand):
            return ("[⚠ 한글 추출이 깨진 것 같습니다. 이 PDF는 폰트 인코딩 문제로 "
                    "텍스트가 손상되었습니다. 원본 txt/markdown 또는 docx로 넣어주세요.]\n\n"
                    + cand)
        return cand
    except Exception:
        pass
    if pypdf_text.strip():
        if _looks_garbled(pypdf_text):
            return ("[⚠ 한글 추출이 깨진 것 같습니다. 원본 txt/markdown/docx로 "
                    "넣어주세요.]\n\n" + pypdf_text)
        return pypdf_text
    raise RuntimeError("pdf 추출 라이브러리가 필요해요: pip install pypdf")


def _extract_docx(raw: bytes) -> str:
    try:
        import docx  # python-docx
    except Exception:
        raise RuntimeError("python-docx가 필요해요: pip install python-docx")
    d = docx.Document(io.BytesIO(raw))
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


# ── 학생 그림: 이미지 → 쉬운 말 설명 (GPT-4V 필요) ────────
def image_to_easy_text(filename: str, raw: bytes,
                       api_key: str, model: str = "gpt-4o") -> str:
    """
    학생 그림을 GPT-4V로 해석해, 학생이 쓸 법한 '쉬운 한국어 단어/짧은 말'로 설명.
    (캐릭터 일관성 추출이 아니라, '무엇을 어떤 쉬운 말로 표현했나'에 초점)
    PDF면 첫 페이지를 이미지로 렌더링해서 보냄.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    name = filename.lower()
    if name.endswith(".pdf"):
        img_bytes, mime = _pdf_first_page_png(raw), "image/png"
    else:
        ext = name.split(".")[-1]
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
        img_bytes = raw

    b64 = base64.b64encode(img_bytes).decode()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text":
                    "이 그림에 무엇이 그려져 있는지 아주 쉬운 한국어로 설명해줘.\n"
                    "- 발달장애 학생이 쓸 법한 짧고 쉬운 말과 단어로.\n"
                    "- 한 줄에 하나씩, 짧은 문장으로.\n"
                    "- 그림에 보이는 것만. 어려운 낱말, 추측, 전문용어는 쓰지 마.\n"
                    "- 예) 노란 버스가 있어요 / 사람이 손을 들어요"}
            ]
        }],
    )
    return resp.choices[0].message.content.strip()


def _pdf_first_page_png(raw: bytes) -> bytes:
    try:
        import pdfplumber
    except Exception:
        raise RuntimeError("pdfplumber가 필요해요: pip install pdfplumber")
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        page = pdf.pages[0]
        im = page.to_image(resolution=150).original
        buf = io.BytesIO(); im.save(buf, format="PNG")
        return buf.getvalue()
