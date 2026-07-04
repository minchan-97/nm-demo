"""
app.py — "판정=설명" 끝판왕 데모

같은 자료(PDF/docx/txt)를 LLM과 NM에 모두 준다.
LLM이 답하면 → 각 문장 밑에 NM이 색과 주석으로 '근거 있나/벗어났나'를 판정.
판정(색)과 설명(logP·근거)이 한 몸. 사후 설명이 아니라 판정=설명.

실행: streamlit run app.py
"""
import sys, os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
from verify_core import DocumentVerifier

st.set_page_config(page_title="판정=설명 데모", layout="wide")

st.title("🔍 판정 = 설명")
st.caption("같은 자료를 LLM과 NM에 함께 준다. LLM 답변의 각 문장을 NM이 "
           "'자료에 근거하나'로 판정하고, **판정(색)과 근거(logP)를 한 몸으로** 보여준다. "
           "사후 설명이 아니라 — 판정하는 계산이 곧 설명이다.")

# ── 상태 ────────────────────────────────────────────────────
if "verifier" not in st.session_state:
    st.session_state.verifier = DocumentVerifier()
verifier = st.session_state.verifier

col_left, col_right = st.columns([1, 1])

# ── 왼쪽: 자료 투입 + 학습 ──────────────────────────────────
with col_left:
    st.subheader("1. 자료 (LLM과 NM에 동일 투입)")
    up = st.file_uploader("근거 자료 (pdf / docx / txt)", type=["pdf", "docx", "txt"])
    manual = st.text_area("또는 직접 붙여넣기", height=180,
                          placeholder="이 자료가 LLM의 답변 근거이자, NM의 판정 기준이 됩니다.")

    corpus_text = ""
    if up is not None:
        try:
            from file_ingest import extract_text
            corpus_text = extract_text(up.name, up.read())
            if corpus_text.strip():
                st.success(f"자료 추출: {len(corpus_text)}자")
            else:
                st.warning("텍스트가 추출되지 않았습니다(스캔 PDF일 수 있음).")
        except Exception as e:
            st.error(f"추출 실패: {e}")
    elif manual.strip():
        corpus_text = manual.strip()

    if corpus_text and st.button("NM 학습(자료 기준 세우기)", type="primary"):
        with st.spinner("자료로 NM 학습 중(CPU, 로컬)..."):
            verifier.train_on_document(corpus_text)
            st.session_state.corpus_text = corpus_text
        st.success("NM 학습 완료 — 이 자료가 판정 기준입니다.")

# ── 오른쪽: LLM 답변 + 검증 ─────────────────────────────────
with col_right:
    st.subheader("2. LLM 답변 검증")
    if not verifier.trained:
        st.info("먼저 왼쪽에서 자료를 넣고 NM을 학습하세요.")
    else:
        st.caption("LLM에게 같은 자료로 질문해 얻은 답변을 붙여넣으세요. "
                   "(또는 아래에서 직접 OpenAI 호출)")

        # OpenAI 직접 호출(선택)
        with st.expander("🤖 OpenAI로 답변 생성(선택)"):
            okey = st.text_input("OpenAI Key", type="password")
            question = st.text_input("질문", placeholder="이 자료의 핵심 내용을 요약해줘")
            if st.button("답변 생성") and okey and question:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=okey)
                    ctx = st.session_state.get("corpus_text", "")[:4000]
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user",
                                   "content": f"다음 자료를 참고해 답하라:\n{ctx}\n\n질문: {question}"}],
                        temperature=0.5)
                    st.session_state.llm_answer = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"생성 실패: {e}")

        answer = st.text_area("LLM 답변", height=160,
                              value=st.session_state.get("llm_answer", ""))

        if st.button("문장별 검증", type="primary") and answer.strip():
            results = verifier.verify_answer(answer)
            st.session_state.results = results

# ── 하단: 검증 결과 (문장별 색 + 주석) ──────────────────────
results = st.session_state.get("results")
if results:
    st.markdown("---")
    st.subheader("3. 판정 = 설명")

    summ = verifier.summary(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 문장", summ["총_문장"])
    c2.metric("✅ 근거 있음", summ["근거있음"])
    c3.metric("🟡 경계", summ["경계"])
    c4.metric("❌ 벗어남", summ["벗어남"])

    st.write("")
    color_map = {
        "green": ("#e6f4ea", "#137333", "✅"),
        "orange": ("#fef7e0", "#b06000", "🟡"),
        "red": ("#fce8e6", "#c5221f", "❌"),
        "gray": ("#f1f3f4", "#5f6368", "⬜"),
    }
    for r in results:
        bg, fg, icon = color_map[r["color"]]
        html = f"""
        <div style="background:{bg}; border-left:5px solid {fg};
                    padding:10px 14px; margin:6px 0; border-radius:4px;">
          <div style="color:#202124; font-size:15px;">{r['sentence']}</div>
          <div style="color:{fg}; font-size:12px; margin-top:5px;">
             {icon} {r['note']}
          </div>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)

    st.caption("초록=자료에 근거함 · 노랑=경계(일부 이탈) · 빨강=자료에서 벗어남(환각 가능). "
               "각 판정 옆 logP가 곧 근거 — 판정과 설명이 분리되지 않는다.")

