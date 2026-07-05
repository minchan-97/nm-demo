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

# ── Kiwi 형태소 분석 상태 표시 ──────────────────────────────
try:
    from korean_tokenizer_kiwi import kiwi_available
    if kiwi_available():
        st.success("✅ Kiwi 형태소 분석 켜짐 — 어간 추출로 어미 변화에 강건 "
                   "('간다'와 '갑니다'를 같은 어간으로 인식)", icon="✅")
    else:
        st.warning("⚠️ Kiwi 미설치 — 규칙 토크나이저로 동작(어미 문제 잔존). "
                   "requirements.txt에 kiwipiepy 포함 여부/설치 로그 확인.", icon="⚠️")
except Exception as _e:
    st.warning(f"⚠️ 토크나이저 상태 확인 실패: {_e}")

# ── 상태 ────────────────────────────────────────────────────
if "verifier" not in st.session_state:
    st.session_state.verifier = DocumentVerifier()
verifier = st.session_state.verifier

# 사이드바: OpenAI 키(있으면 claim 추출에 LLM 사용) + 세션 저장 폴더
with st.sidebar:
    st.markdown("### 설정")
    openai_key = st.text_input("OpenAI Key (선택)", type="password",
                               help="입력 시 항목-값 추출을 LLM으로. 없으면 규칙 추출.")
    st.session_state.openai_key = openai_key

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
            # OpenAI 키 있으면 LLM 추출기 사용(claim 추출), 없으면 규칙 추출
            llm_extractor = None
            key = st.session_state.get("openai_key", "")
            if key:
                try:
                    from llm_extractor import make_openai_extractor
                    llm_extractor = make_openai_extractor(key)
                except Exception as _e:
                    st.warning(f"LLM 추출기 준비 실패(규칙 추출로 진행): {_e}")
            verifier.train_on_document(corpus_text, llm_extractor=llm_extractor)
            st.session_state.corpus_text = corpus_text
        mode = "LLM 추출" if llm_extractor else "규칙 추출"
        st.success(f"NM 학습 완료 — 판정 기준 확립. 항목-값 추출: {mode}")

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
            okey = st.session_state.get("openai_key", "")
            if not okey:
                st.caption("사이드바에 OpenAI Key를 넣으면 사용할 수 있어요.")
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
            st.session_state.answer = answer

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

    # ── 층2: 항목-값 사실 검증 (claim) ──────────────────────
    st.markdown("---")
    st.subheader("4. 항목-값 사실 검증")
    st.caption("NM(문장 통계)이 놓치는 '자료에 있는 단어의 틀린 조합'을 잡는다. "
               "예: 자료엔 '수학여행 10월'인데 답변이 '수학여행 9월'이면, "
               "단어는 다 자료에 있어 NM은 통과시키지만 여기서 잡힌다.")
    try:
        claim_results = verifier.verify_claims(answer)
    except Exception as _e:
        claim_results = []
        st.info(f"항목-값 검증을 실행하지 못했습니다: {_e}")

    if not claim_results:
        st.write("검출된 항목-값 주장이 없습니다.")
    else:
        cmap = {"green": ("#e6f4ea", "#137333", "✅"),
                "red": ("#fce8e6", "#c5221f", "❌"),
                "gray": ("#f1f3f4", "#5f6368", "⬜")}
        for r in claim_results:
            bg, fg, icon = cmap.get(r["color"], cmap["gray"])
            path_html = "<br>".join(f"· {p}" for p in r.get("path", []))
            html = f"""
            <div style="background:{bg}; border-left:5px solid {fg};
                        padding:10px 14px; margin:6px 0; border-radius:4px;">
              <div style="color:#202124; font-size:15px;">
                 {icon} <b>{r['item']}</b> = {r['value']}
              </div>
              <div style="color:{fg}; font-size:13px; margin-top:4px;">{r['reason']}</div>
              <div style="color:#5f6368; font-size:11px; margin-top:5px;">{path_html}</div>
            </div>"""
            st.markdown(html, unsafe_allow_html=True)
        st.caption("각 판정의 경로(· 로 표시)가 곧 설명이다. "
                   "LLM이 후보를 뽑고(scribe), 자료 원문이 심판하고, 대조는 투명하다.")

    # ── 세션 저장/불러오기 (모바일 대응 · 문제점 파악용) ────
    st.markdown("---")
    st.subheader("5. 세션 저장 (문제점 파악용)")
    st.caption("검증 결과를 pkl로 폰에 저장해두면, 나중에 다시 올려서 "
               "'어디서 오탐/미탐했나'를 되짚을 수 있어요. 실제 케이스가 쌓입니다.")

    import io as _io, pickle as _pk
    from session_store import VerificationSession, analyze_problems
    from dataclasses import asdict as _asdict

    fname = st.text_input("저장 파일명", value="session1.pkl")
    try:
        sess = VerificationSession(
            corpus_preview=st.session_state.get("corpus_text", "")[:300],
            answer_text=st.session_state.get("answer", ""),
            nm_results=results,
            claim_results=claim_results,
            meta={"saved_by": "app"},
        )
        # pkl을 메모리에 만들어 다운로드 버튼으로 (모바일=폰에 저장됨)
        buf = _io.BytesIO()
        _pk.dump(_asdict(sess), buf)
        buf.seek(0)
        st.download_button(
            "⬇ 이 세션 저장 (폰에 다운로드)",
            data=buf.getvalue(),
            file_name=fname if fname.endswith(".pkl") else fname + ".pkl",
            mime="application/octet-stream",
        )
    except Exception as _e:
        st.error(f"세션 준비 실패: {_e}")

    # ── 저장한 세션 다시 불러와 분석 ──
    st.markdown("**저장한 세션 다시 분석**")
    up = st.file_uploader("세션 pkl 불러오기", key="sess_load",
                          help="이전에 저장한 세션 pkl을 올리면 오탐/미탐을 집계합니다.")
    if up is not None:
        if not up.name.lower().endswith(".pkl"):
            st.warning("`.pkl` 파일만 불러올 수 있어요. (선택: " + up.name + ")")
        else:
            try:
                loaded = _pk.load(_io.BytesIO(up.getvalue()))
                st.caption(f"불러옴: {up.name} · 답변: "
                           f"{(loaded.get('answer_text','') or '')[:40]}")
                st.write(f"NM 판정 {len(loaded.get('nm_results', []))}건 · "
                         f"claim 판정 {len(loaded.get('claim_results', []))}건 · "
                         f"피드백 {len(loaded.get('feedback', {}))}건")
                prob = analyze_problems(loaded)
                st.write("**문제점 집계:**", prob)
                # claim 판정 다시 보여주기(오탐 확인용)
                with st.expander("이 세션의 claim 판정 보기"):
                    for i, r in enumerate(loaded.get("claim_results", [])):
                        mk = {"green": "✅", "red": "❌", "gray": "⬜"}.get(r.get("color"), "•")
                        st.write(f"{mk} [{r.get('item')}={r.get('value')}] {r.get('reason','')[:60]}")
            except Exception as _e:
                st.error(f"불러오기 실패: {_e}")
