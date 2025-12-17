import streamlit as st
import re
import os
import time
from dotenv import load_dotenv

# [Import] src 폴더 내의 모듈을 불러옵니다.
# 실제 파일 경로에 맞게 수정 필요 (예: from src.toxic_detector import ...)
from toxic_detector import ToxicClauseDetector
from ollama_detctor import ToxicClauseDetectorOllama
from llm_service import LLM_gemini

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="SafeSign - On-Device AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 헬퍼 함수들 ---

def extract_text_from_pdf(pdf_file, api_key): 
    """
    [OCR] PDF 텍스트 추출은 성능이 좋은 Gemini Vision을 사용합니다.
    (Ollama Llama3는 Vision 기능이 없거나 약하기 때문)
    """
    try:
        pdf_file_bytes = pdf_file.read()   
        # Gemini 1.5 Flash가 OCR 가성비가 좋음
        gemini = LLM_gemini(gemini_api_key=api_key, model='gemini-2.5-flash')
        result = gemini.pdf_to_text(pdf_file_bytes)
        return result
    except Exception as e:
        st.error(f"PDF 처리 중 오류 발생: {e}")
        return None

def get_dummy_contract_text():
    return """
제1조 (목적) 본 계약은 사용자 (주)악덕상사(이하 "갑")와 근로자 홍길동(이하 "을")의 근로조건을 정함을 목적으로 한다.
제2조 (근로시간) 근로시간은 09:00부터 18:00까지로 한다. "갑"은 업무상 필요한 경우 "을"에게 연장근로를 명할 수 있으며 "을"은 이에 포괄적으로 동의한다.
제3조 (임금) 월 급여는 250만원으로 하며, 이는 연장/야간/휴일 근로수당을 모두 포함한 포괄임금으로 한다.
제4조 (퇴직금) 1년 미만 근무 시 퇴직금은 지급하지 않으며, 퇴사 시 교육비 명목으로 300만원을 배상한다.
제5조 (해고) "갑"은 "을"의 업무 성과가 저조하다고 판단될 경우 즉시 해고할 수 있다.
"""

def parse_text_to_chunks(text):
    if not text: return []
    split_pattern = r'(?=\n\s*제\s*\d+\s*조)'
    chunks = re.split(split_pattern, text)
    return [c.strip() for c in chunks if len(c.strip()) > 10]

# --- 3. 메인 어플리케이션 --- 
def main():
    # 사이드바 설정
    with st.sidebar:
        st.title("🛡️ SafeSign Local")
        st.caption("On-Device Analysis with Ollama")
        st.markdown("---")
        
        load_dotenv()
        env_key = os.getenv("GEMINI_API_KEY")
        
        # 1. OCR용 API 키 (필수 아님, 없으면 더미 사용)
        api_key_input = st.text_input(
            "Gemini API Key (OCR용)", 
            value=env_key if env_key else "", 
            type="password",
            help="PDF 이미지 인식을 위해 사용됩니다."
        )
        
        # 2. Ollama 모델 선택
        ollama_model = st.selectbox(
            "Ollama Model", 
            ["llama3", "mistral", "gemma", "hf.co/LiquidAI/LFM2-8B-A1B-GGUF:Q4_K_M"],
            index=0
        )

        st.info(f"💡 분석 엔진: Local {ollama_model}\n(내 컴퓨터의 GPU/CPU를 사용합니다)")

    # 메인 화면
    st.title("📄 근로계약서 독소조항 판별기 (Local Ver.)")
    st.markdown("보안이 중요한 계약서, **외부 서버 전송 없이** 내 컴퓨터의 Ollama가 직접 분석합니다.")

    # 파일 업로드 및 텍스트 로딩
    uploaded_file = st.file_uploader("근로계약서 PDF 업로드", type=["pdf"])
    
    contract_content = ""
    
    if uploaded_file is not None:
        if api_key_input:
            with st.spinner("👀 Gemini가 문서를 읽고 있습니다 (OCR)..."):
                extracted_text = extract_text_from_pdf(uploaded_file, api_key_input)
                if extracted_text:
                    contract_content = extracted_text
                    st.success("텍스트 추출 완료!")
                else:
                    contract_content = get_dummy_contract_text()
                    st.warning("추출 실패. 예시 데이터를 사용합니다.")
        else:
            st.warning("OCR용 API 키가 없어 예시 데이터를 사용합니다.")
            contract_content = get_dummy_contract_text()
    else:
        contract_content = get_dummy_contract_text()

    # 텍스트 에디터
    final_text = st.text_area("분석 대상 텍스트", value=contract_content, height=300)

    # [분석 버튼]
    if st.button("🚀 로컬 AI 분석 시작", use_container_width=True):
        
        # 1. Parsing
        chunks = parse_text_to_chunks(final_text)
        if not chunks:
            st.error("분석할 조항을 찾지 못했습니다.")
            st.stop()

        # 2. Detector 초기화 (Ollama)
        # 로컬 모델 로딩은 시간이 걸리므로 캐싱 필수
        @st.cache_resource
        def get_ollama_detector(model_name):
            return ToxicClauseDetectorOllama(model_name=model_name)
        
        with st.spinner(f"⚙️ Ollama({ollama_model}) 모델 및 법률 DB 로딩 중..."):
            try:
                detector = get_ollama_detector(ollama_model)
            except Exception as e:
                st.error(f"Ollama 연결 실패: {e}")
                st.info("터미널에서 'ollama serve'가 켜져 있는지 확인하세요.")
                st.stop()

        st.info(f"총 {len(chunks)}개의 조항을 분석합니다. (로컬 모델 특성상 시간이 소요될 수 있습니다)")

        # 3. 분석 실행 (순차 처리 권장)
        # Ollama는 동시 요청 처리가 약하므로 max_concurrent=1 설정
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_results = []
        
        try:
            # detect 함수가 리스트를 받아 내부적으로 처리함
            raw_results = detector.detect(chunks, max_concurrent=1)
            
            # 결과 가공 및 제안 생성
            for i, res in enumerate(raw_results):
                res['id'] = i + 1
                res['suggestion'] = ""
                
                # [수정] 독소조항인 경우 자동으로 제안(쉬운 해석) 생성
                if res['is_toxic']:
                    status_text.text(f"⚠️ 제{i+1}조 분석 중... (개선안 생성 포함)")
                    try:
                        res['suggestion'] = detector.generate_easy_suggestion(res)
                    except Exception:
                        res['suggestion'] = "제안 생성 실패"
                
                processed_results.append(res)
                progress_bar.progress((i + 1) / len(chunks))
                
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            st.stop()

        status_text.empty()
        st.success("✅ 분석 완료!")
        
        # 4. 결과 리포트 출력
        st.divider()
        
        toxic_indices = [i for i, r in enumerate(processed_results) if r['is_toxic']]
        
        col1, col2 = st.columns(2)
        col1.metric("분석된 조항", f"{len(chunks)}건")
        col2.metric("위험 조항 발견", f"{len(toxic_indices)}건", delta="-주의" if toxic_indices else "안전")

        # 상세 결과 탭
        tab1, tab2 = st.tabs(["🚨 위험 조항 리포트", "📑 전체 조항 보기"])
        
        with tab1:
            if not toxic_indices:
                st.balloons()
                st.success("발견된 독소조항이 없습니다!")
            else:
                for idx in toxic_indices:
                    res = processed_results[idx]
                    
                    with st.expander(f"⚠️ [위험] 제{res['id']}조 (위험도: {res['risk_score']})", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("❌ 원문")
                            st.error(res['clause'])
                            st.markdown(f"**🔍 판단 근거:**\n{res['reason']}")
                        with c2:
                            st.caption("💡 AI 수정 제안 & 쉬운 해석")
                            # [수정] 버튼 없이 바로 내용 표시
                            if res['suggestion']:
                                st.markdown(res['suggestion'])
                            else:
                                st.info("제안 내용을 생성하지 못했습니다.")
                            
                            with st.popover("참고 법령 보기"):
                                st.text(res['context_used'])
        
        with tab2:
            for res in processed_results:
                icon = "🔴" if res['is_toxic'] else "🟢"
                with st.expander(f"{icon} 제{res['id']}조"):
                    st.write(res['clause'])
                    st.caption(f"판단: {res['reason']}")

if __name__ == "__main__":
    main()