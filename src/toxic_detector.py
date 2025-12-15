import os
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval import Rubric

# [Import]
from llm_service import LLM_gemini
from law.legal_context import LawContextManager
from law.precedent_context import PrecedentContextManager

#load_dotenv()

# --- 1. DeepEval용 Gemini 어댑터 ---
class GeminiDeepEvalAdapter(DeepEvalBaseLLM):
    def __init__(self, llm_service: LLM_gemini):
        self.llm_service = llm_service
        self.model_name = llm_service.model_name

    def load_model(self):
        return self.llm_service.client

    def generate(self, prompt: str) -> str:
        response = self.llm_service.generate(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. 독소조항 판별기 클래스 ---
class ToxicClauseDetector:
    def __init__(self, api_key=None):
        print("🛡️ ToxicClauseDetector (Pro Model) 초기화 중...")
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API Key가 없습니다.")

        
        self.llm_service = LLM_gemini(gemini_api_key=api_key, model="gemini-2.5-flash")
        self.evaluator_llm = GeminiDeepEvalAdapter(self.llm_service)
        
        # DB 매니저
        self.law_manager = LawContextManager()
        self.precedent_manager = PrecedentContextManager()
        self.law_manager.initialize_database()
        self.precedent_manager.initialize_database()

       # 1. 3대 원칙(Legality, Fairness, Clarity)을 핵심 기준으로 설정
        # 점수가 높을수록 '위험(Toxic)'한 것으로 기준을 뒤집습니다.
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
        # 2. 개선된 채점 루브릭 (Rubric)
        # - 점수대별로 '명백한 불법'과 '해석상 불리함'을 확실히 구분
        self.rubric = [
            Rubric(score_range=(0, 2), expected_outcome="3대 원칙(효력, 공정성, 명확성)을 모두 충족하는 완벽한 조항."),
            Rubric(score_range=(3, 5), expected_outcome="[명확성 부족] - 법적으로 문제는 없으나, 표현이 모호하여 회사의 자의적 해석이 우려됨."),
            Rubric(score_range=(6, 8), expected_outcome="[공정성 결여] - 불법 직전의 수준. 근로자에게 일방적으로 불리하거나 입증 책임을 전가함."),
            Rubric(score_range=(9, 10), expected_outcome="[법적 효력 없음] - 근로기준법 강행규정 위반으로 해당 조항 자체가 무효임."),
        ]

        self.evaluation_steps = [
            "0단계 [의도 파악]: 조항의 핵심 의도(임금 삭감, 해고 용이성, 책임 전가 등)를 먼저 파악하고, 일반적인 법률 지식을 로딩한다.",
            "1단계 [Legality/치명적]: 근로기준법 강행규정 위반 여부를 최우선 확인한다. 특히 '퇴직금 포기', '손해배상액 예정', '강제 근로' 키워드가 있으면 즉시 10점을 부여한다.",
            "2단계 [Fairness/위험]: 불법이 아니라면, 권리와 의무의 균형을 본다. 회사에만 유리하거나 근로자에게 과도한 의무를 부과하면 6~8점을 부여한다.",
            "3단계 [Clarity/주의]: 내용이 공정해 보여도, '기타', '상당한' 등 자의적 해석이 가능한 모호한 단어가 있다면 3~5점을 부여한다.",
            "4단계 [종합 판단]: 위 단계들을 거쳐 점수를 매기되, 법적 근거가 확실하지 않은 회색지대라면 근로자에게 불리한 쪽(보수적)으로 해석하여 최종 점수를 확정한다."
        ]

    def _retrieve_context(self, clause_text):
        # 1. 법령 검색
        laws = self.law_manager.search_relevant_laws(clause_text, k=2)
        law_text = "\n".join(laws) if laws else "관련 법령 검색 결과 없음 (일반 법률 지식으로 판단 요망)"

        # 2. 판례 검색
        precedents = self.precedent_manager.search_relevant_precedents(clause_text, k=1)
        precedent_text = precedents[0] if precedents else "관련 판례 검색 결과 없음"

        return f"=== [관련 법령] ===\n{law_text}\n\n=== [관련 판례] ===\n{precedent_text}"

    def detect(self, clause_text):
        # print(f"🕵️ 조항 분석 중: {clause_text[:30]}...")
        
        retrieved_context = self._retrieve_context(clause_text)
        
        # G-Eval 평가
        toxic_metric = GEval(
            name="Toxicity Score", # 이름 변경
            criteria=self.toxic_criteria,
            rubric=self.rubric,
            evaluation_steps=self.evaluation_steps,
            model=self.evaluator_llm, 
            threshold=5, # 5점 이상이면 독소조항으로 간주
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        test_case = LLMTestCase(
            input=clause_text,
            actual_output="평가 대상",
            retrieval_context=[retrieved_context]
        )

        toxic_metric.measure(test_case)
        
        # 이제 점수(0~10)가 곧 위험도입니다. 뒤집을 필요가 없습니다.
        risk_score = toxic_metric.score # 0~10점 (DeepEval 버전에 따라 0~1일 수도 있음, 아래 보정)
        
        # DeepEval이 0~1 사이 값을 리턴하는 경우 10을 곱해줌
        if risk_score <= 1.0:
            risk_score *= 10
            
        # 4점 이상이면 독소조항 (기준 강화)
        is_toxic = risk_score >= 4.0
        
        # 디버깅용 출력 (터미널에서 확인 가능)
        print(f"[{'🚨위험' if is_toxic else '✅안전'}] 점수: {risk_score} | 내용: {clause_text[:20]}...")

        return {
            "clause": clause_text,
            "is_toxic": is_toxic,
            "risk_score": round(risk_score, 1),
            "reason": toxic_metric.reason,
            "context_used": retrieved_context
        }

    def generate_easy_suggestion(self, detection_result):
        if not detection_result['is_toxic']:
            return "✅ **안전한 조항입니다.**"

        prompt = f"""
        당신은 근로자 편인 법률 전문가입니다. 다음 독소조항을 분석하세요.
        
        [원문]: {detection_result['clause']}
        [이유]: {detection_result['reason']}
        [근거]: {detection_result['context_used']}

        다음 두 가지를 마크다운으로 작성:
        1. **⚠️ 쉬운 해석**: 근로자가 이해하기 쉽게 1~2문장으로 '왜 위험한지' 설명 및 요약.
        2. **💡 수정 제안**: 법에 맞는 공정한 조항 예시.
        """
        response = self.llm_service.generate(prompt)
        return response.text