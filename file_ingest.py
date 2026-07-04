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


def _extract_pdf(raw: bytes) -> str:
    try:
        import pdfplumber
    except Exception:
        raise RuntimeError("pdfplumber가 필요해요: pip install pdfplumber")
    out = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            out.append(t)
    return "\n".join(out)


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
