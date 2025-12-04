import streamlit as st
import time
import pandas as pd
from datetime import datetime

# ==========================================
# [설정] 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="근로계약서 독소조항 판별기 (AI Guardian)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# [Role C, B, A] Backend Mock Modules
# 실제 개발 시 src/ 폴더의 모듈을 import 해야 함
# ==========================================

# [Role C] src.parser.pdf_parser & text_chunker
def mock_parser(uploaded_file):
    """
    [TODO: Role C] 실제 PDF 파싱 및 Regex Chunking 로직 구현
    """
    time.sleep(1) # 처리 시간 시뮬레이션
    return [
        "제1조 (목적) 본 계약은 사용자와 근로자의 근로조건을 정함을 목적으로 한다.",
        "제2조 (임금) 월 급여는 200만원으로 하며, 이는 연장근로수당을 포함한 포괄임금으로 한다.", # 독소조항 예시
        "제3조 (근로시간) 근로시간은 09:00부터 18:00까지로 한다.",
        "제4조 (퇴직금) 1년 미만 근무 후 퇴사 시 퇴직금은 지급하지 않으며 손해배상을 청구한다." # 독소조항 예시
    ]

# [Role C] src.retriever.law_api & case_search
def mock_retriever(clause_text):
    """
    [TODO: Role C] 법제처 API 및 HuggingFace Vector DB 검색 구현
    """
    return {
        "law": "근로기준법 제N조...",
        "case": "대법원 20XX다XXXX 판결..."
    }

# [Role B] src.evaluator.g_eval & faithfulness
def mock_evaluator(clause, context):
    """
    [TODO: Role B] DeepEval G-Eval 및 Faithfulness Metric 구현
    """
    # 독소조항 시뮬레이션 (특정 키워드로 구분)
    if "포괄임금" in clause or "손해배상" in clause:
        return {
            "score": 8, # 위험도 1~10
            "is_toxic": True,
            "reason": "포괄임금제 오남용 및 위약금 예정 금지 조항 위반 가능성이 높음.",
            "faithfulness": 0.95
        }
    else:
        return {
            "score": 1,
            "is_toxic": False,
            "reason": "법적 문제 없음.",
            "faithfulness": 1.0
        }

# [Role A] src.generator.report_gen
def mock_generator(clause, evaluation):
    """
    [TODO: Role A] LLM을 이용한 쉬운 해석 및 수정 제안 생성
    """
    if evaluation["is_toxic"]:
        return "이 조항은 당신이 야근을 해도 추가 수당을 받기 어렵게 만들 수 있어요. '포괄임금'이라는 단어를 주의하세요."
    return "표준적인 근로계약 조항입니다. 안심하셔도 됩니다."

# ==========================================
# [Role D] Frontend UI Logic
# ==========================================

def main():
    # 1. 사이드바: 설정 및 파일 업로드
    with st.sidebar:
        st.title("⚖️ AI Contract Guardian")
        st.markdown("---")
        
        st.subheader("1. 설정")
        api_key = st.text_input("OpenAI/Gemini API Key", type="password")
        
        st.subheader("2. 계약서 업로드")
        uploaded_file = st.file_uploader("근로계약서(PDF/IMG)를 올려주세요", type=["pdf", "png", "jpg"])
        
        st.markdown("---")
        st.info("💡 이 도구는 법적 효력이 없으며 참고용으로만 사용하세요.")

    # 2. 메인 화면: 헤더
    st.title("📄 근로계약서 독소조항 판별기")
    st.markdown("""
    **RAG와 DeepEval**을 활용하여 계약서 내 숨겨진 **독소조항(Toxic Clause)**을 찾아내고, 
    이해하기 쉬운 **해설**을 제공합니다.
    """)

    # 3. 분석 로직 실행
    if uploaded_file is not None:
        st.success("파일 업로드 완료! 분석을 시작합니다.")
        
        if st.button("🚀 독소조항 분석 시작", use_container_width=True):
            
            # [Step 1] Parsing
            with st.status("🔍 계약서를 읽고 조항을 나누는 중...", expanded=True) as status:
                st.write("텍스트 추출 중...")
                chunks = mock_parser(uploaded_file)
                st.write(f"총 {len(chunks)}개의 조항이 식별되었습니다.")
                time.sleep(0.5)
                
                # 결과 저장용 리스트
                results = []
                
                # [Step 2] Analysis Loop (Progress Bar)
                progress_bar = st.progress(0)
                
                for i, clause in enumerate(chunks):
                    # UI 업데이트
                    status.update(label=f"판별 중... ({i+1}/{len(chunks)}): 제{i+1}조 분석", state="running")
                    
                    # RAG & DeepEval Pipeline Execution
                    context = mock_retriever(clause)
                    eval_result = mock_evaluator(clause, context)
                    easy_explanation = mock_generator(clause, eval_result)
                    
                    results.append({
                        "id": i+1,
                        "clause": clause,
                        "score": eval_result["score"],
                        "is_toxic": eval_result["is_toxic"],
                        "reason": eval_result["reason"],
                        "explanation": easy_explanation,
                        "faithfulness": eval_result["faithfulness"]
                    })
                    
                    # 진행률 업데이트
                    progress_bar.progress((i + 1) / len(chunks))
                    time.sleep(0.5) # 실제 속도에 맞춰 조정

                status.update(label="✅ 분석 완료!", state="complete", expanded=False)

            # 4. 결과 대시보드 (Session State에 저장하여 리렌더링 방지 가능)
            st.divider()
            st.subheader("📊 분석 리포트")

            # 요약 지표
            toxic_count = sum(1 for r in results if r["is_toxic"])
            col1, col2, col3 = st.columns(3)
            col1.metric("총 조항 수", f"{len(chunks)}개")
            col2.metric("발견된 독소조항", f"{toxic_count}개", delta="-위험" if toxic_count > 0 else "안전")
            col3.metric("평균 신뢰도(Faithfulness)", "0.98")

            # 상세 결과 뷰어
            st.markdown("### 📝 상세 조항 분석")
            
            tab1, tab2 = st.tabs(["🚨 위험 조항만 보기", "📑 전체 조항 보기"])
            
            with tab1:
                if toxic_count == 0:
                    st.success("독소조항이 발견되지 않았습니다!")
                else:
                    for res in results:
                        if res["is_toxic"]:
                            with st.expander(f"⚠️ [위험] 제{res['id']}조 분석 결과 (위험도: {res['score']}/10)", expanded=True):
                                st.markdown(f"**원문:**\n> {res['clause']}")
                                st.error(f"**판단 근거:** {res['reason']}")
                                st.info(f"**💡 쉬운 해석:** {res['explanation']}")
                                st.caption(f"AI 신뢰도 검증: {res['faithfulness']}")

            with tab2:
                for res in results:
                    icon = "🔴" if res['is_toxic'] else "🟢"
                    title = f"{icon} 제{res['id']}조"
                    with st.expander(title):
                        st.write(res['clause'])
                        if res['is_toxic']:
                             st.warning(res['explanation'])
                        else:
                             st.success("안전한 조항입니다.")

    else:
        # 파일이 없을 때 보여줄 안내 화면
        st.warning("왼쪽 사이드바에서 계약서 파일을 업로드해주세요.")
        #  
        # (실제 프로젝트에서는 assets/sample.png 이미지를 로드)

if __name__ == "__main__":
    main()