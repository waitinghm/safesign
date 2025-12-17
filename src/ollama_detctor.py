import os
import time
from typing import List, Dict, Union
from dotenv import load_dotenv

# Ollama & DeepEval Imports
import ollama
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric
from deepeval.evaluate import AsyncConfig

# Project Modules
from law.legal_context import LawContextManager
from law.precedent_context import PrecedentContextManager

load_dotenv()

# --- 1. DeepEval용 Ollama 어댑터 (ollama.chat 사용) ---
class OllamaDeepEvalAdapter(DeepEvalBaseLLM):
    """
    DeepEval 프레임워크가 Ollama(Local LLM)를 인식하고 제어할 수 있도록 돕는 어댑터.
    LangChain을 거치지 않고 공식 ollama 라이브러리의 chat 함수를 직접 사용합니다.
    """
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        # ollama 라이브러리는 별도 클라이언트 객체 생성 불필요

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        """
        공식 ollama.chat 함수를 사용하여 답변을 생성합니다.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=False
            )
            return response['message']['content']
        except Exception as e:
            return f"Ollama Generation Error: {e}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. 독소조항 판별기 (Ollama 버전) ---
class ToxicClauseDetectorOllama:
    def __init__(self, model_name="llama3"):
        print(f"🛡️ ToxicClauseDetector (Ollama: {model_name}) 초기화 중...")
        
        # Ollama 어댑터 연결
        self.evaluator_llm = OllamaDeepEvalAdapter(model_name=model_name)
        
        # DB 매니저 초기화 (RAG)
        self.law_manager = LawContextManager()
        self.law_manager.initialize_database()
        
        self.precedent_manager = PrecedentContextManager()
        self.precedent_manager.initialize_database()

        # [평가 기준] - Gemini 버전과 동일
        self.toxic_criteria = """
        당신은 근로계약서의 [법적 효력], [공정성], [명확성]을 심사하는 전문 노무 AI입니다.
        아래 3가지 핵심 기준에 따라 조항을 분석하고 점수를 매기세요.

        [분석 기준 3대 원칙]
        1. 법적 효력 (Legality) - [치명적/Red Zone]
           - 근로기준법 등 강행법규를 위반하는가?
           - 예: 최저임금 미달, 퇴직금 포기 각서, 위약금 예정(손해배상액 명시), 해고 예고 위반.
           - 판단: 위반 시 무조건 9~10점 부여.

        2. 공정성 (Fairness) - [위험/Orange Zone]
           - 사용자(회사)에게만 유리하고 근로자에게 과도한 의무를 부과하는가?
           - 예: "모든 손해를 배상한다(포괄적 배상)", "퇴사 시 후임자를 구해야 한다", "사내 규정 위반 시 무조건 징계".
           - 판단: 불법은 아니더라도 근로자가 억울할 소지가 크면 6~8점 부여.

        3. 명확성 (Clarity) - [주의/Yellow Zone]
           - 자의적 해석이 가능한 모호한 표현이 있는가?
           - 예: "회사가 필요하다고 인정하는 경우", "관례에 따른다", "기타 갑이 정하는 업무".
           - 판단: 문구가 모호하여 분쟁 가능성이 있으면 3~5점 부여.
        """
        
        self.rubric = [
            Rubric(score_range=(0, 2), expected_outcome="3대 원칙(효력, 공정성, 명확성)을 모두 충족하는 완벽한 조항."),
            Rubric(score_range=(3, 5), expected_outcome="[명확성 부족] - 법적으로 문제는 없으나, 표현이 모호하여 회사의 자의적 해석이 우려됨."),
            Rubric(score_range=(6, 8), expected_outcome="[공정성 결여] - 불법 직전의 수준. 근로자에게 일방적으로 불리하거나 입증 책임을 전가함."),
            Rubric(score_range=(9, 10), expected_outcome="[법적 효력 없음] - 근로기준법 강행규정 위반으로 해당 조항 자체가 무효임."),
        ]

        self.evaluation_steps = [
            "1단계 [의도 파악]: 조항의 핵심 의도(임금 삭감, 해고 용이성, 책임 전가 등)를 먼저 파악하고, 일반적인 법률 지식을 로딩한다.",
            "2단계 [Legality/치명적]: 근로기준법 강행규정 위반 여부를 최우선 확인한다. 특히 '퇴직금 포기', '손해배상액 예정', '강제 근로' 키워드가 있으면 즉시 10점을 부여한다.",
            "3단계 [Fairness/위험]: 불법이 아니라면, 권리와 의무의 균형을 본다. 회사에만 유리하거나 근로자에게 과도한 의무를 부과하면 6~8점을 부여한다.",
            "4단계 [Clarity/주의]: 내용이 공정해 보여도, '기타', '상당한' 등 자의적 해석이 가능한 모호한 단어가 있다면 3~5점을 부여한다.",
            "5단계 [종합 판단]: 위 단계들을 거쳐 점수를 매기되, 법적 근거가 확실하지 않은 회색지대라면 근로자에게 불리한 쪽(보수적)으로 해석하여 최종 점수를 확정한다.",
        ]
        
        # G-Eval Metric 객체 생성
        self.toxic_metric = GEval(
            name="Toxicity Score (Ollama)",
            criteria=self.toxic_criteria,
            rubric=self.rubric,
            evaluation_steps=self.evaluation_steps,
            model=self.evaluator_llm, # Ollama가 심사위원
            threshold=5, 
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

    def _retrieve_context(self, clause_text):
        # 1. 법령 검색
        laws = self.law_manager.search_relevant_laws(clause_text, k=2)
        law_text = "\n".join(laws) if laws else "관련 법령 검색 결과 없음"

        # 2. 판례 검색
        precedents = self.precedent_manager.search_relevant_precedents(clause_text, k=1)
        precedent_text = precedents[0] if precedents else "관련 판례 검색 결과 없음"

        return f"=== [관련 법령] ===\n{law_text}\n\n=== [관련 판례] ===\n{precedent_text}"

    def detect(self, clause_texts: List[str], max_concurrent: int = 1) -> List[Dict]:
        """
        Ollama의 JSON 파싱 오류나 불안정성을 고려하여 evaluate 함수 대신 순차적으로 처리합니다.
        """
        print(f"🚀 총 {len(clause_texts)}개 조항에 대한 평가 시작 (Ollama)...")
        
        formatted_results = []
        original_map = {} 

        # 순차 처리 Loop
        for i, text in enumerate(clause_texts):
            print(f"   Processing Clause {i+1}/{len(clause_texts)}...", end="\r")
            
            # 1. RAG 검색
            retrieved_context = self._retrieve_context(text)
            original_map[text] = retrieved_context
            
            # 2. Test Case 생성
            test_case = LLMTestCase(
                input=text,
                actual_output="평가 대상",
                retrieval_context=[retrieved_context]
            )

            # 3. 평가 실행 (Try-Except로 보호)
            try:
                self.toxic_metric.measure(test_case)
                
                # 성공 시 데이터 추출
                metric_score = self.toxic_metric.score
                metric_reason = self.toxic_metric.reason
                
                # 점수 보정 (0.0~1.0 -> 0~10)
                risk_score = metric_score
                if risk_score <= 1.0:
                    risk_score *= 10
                
                is_toxic = risk_score >= 4.0

            except Exception as e:
                # 실패 시
                print(f"\n⚠️ [Skip Clause {i+1}] 모델 응답 오류: {e}")
                risk_score = 0
                is_toxic = False
                metric_reason = f"Ollama 모델 출력 오류 (JSON Parsing Failed): {e}"

            # 결과 저장
            formatted_results.append({
                "clause": text,
                "is_toxic": is_toxic,
                "risk_score": round(risk_score, 1),
                "reason": metric_reason,
                "context_used": retrieved_context
            })

        print("\n✅ 모든 평가가 완료되었습니다.")
        return formatted_results

    def generate_easy_suggestion(self, detection_result):
        """Ollama를 이용해 쉬운 해석 및 수정 제안 생성"""
        if not detection_result['is_toxic']:
            return "✅ **안전한 조항입니다.**"

        prompt = f"""
        당신은 근로자 편인 법률 전문가입니다. 다음 독소조항을 분석하세요.
        
        [원문]: {detection_result['clause']}
        [위험 판단 이유]: {detection_result['reason']}
        [법적 근거]: {detection_result['context_used']}

        다음 두 가지를 마크다운 형식으로 작성해주세요 (한국어):
        1. **⚠️ 쉬운 해석**: 근로자가 이해하기 쉽게 1~2문장으로 요약.
        2. **💡 수정 제안**: 법에 맞는 공정한 조항 예시.
        """
        return self.evaluator_llm.generate(prompt)

# --- 실행 테스트 ---
if __name__ == "__main__":
    try:
        # 테스트할 모델명 설정
        TARGET_MODEL = "hf.co/LiquidAI/LFM2-8B-A1B-GGUF:Q4_K_M"
        
        detector = ToxicClauseDetectorOllama(model_name=TARGET_MODEL)
        
        test_clauses = [
            "퇴사 시 후임자를 구하지 못하면 그로 인한 모든 손해를 배상해야 한다.",
            "수습기간 3개월 동안은 최저임금의 80%만 지급한다."
        ]
        
        # 로컬 모델은 느리므로 max_concurrent를 작게 설정 (실제 로직은 순차 처리됨)
        results = detector.detect(test_clauses, max_concurrent=1)
        
        print("\n" + "="*50)
        for res in results:
            icon = "🚨" if res['is_toxic'] else "✅"
            print(f"{icon} 점수: {res['risk_score']} | 내용: {res['clause'][:30]}...")
            print(f"   이유: {res['reason']}")
            
            if res['is_toxic']:
                print("\n   [AI 제안 생성 중...]")
                suggestion = detector.generate_easy_suggestion(res)
                print(suggestion)
            print("-" * 50)
            
    except Exception as e:
        print(f"오류 발생: {e}")