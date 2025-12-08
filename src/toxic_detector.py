from deepeval.metrics.g_eval import Rubric
from .rag_pipeline import search_precedents
class ToxicClauseDetector():
    def __init__(self):
        # 독소조항 판병하는 llm 모델, DeepEval용 Wrapper 써야함
        self.evaluator_llm = None
        # 법령 정보를 관리하는 인스턴스, DB 초기화 및 법령 검색 기능
        self.law_manager = None
        self.law_manager.initialize_database()
        # 독소조항 판별에 쓰일 평가기준
        self.toxic_criteria = ""
        # 독소조항 판별에 쓰일 rubric
        self.rubric = []
        # 독소조항 판별에 쓰일 CoT
        self.evalutation_step = []

    # 법령 및 판례 검색하는 함수
    def _retrieve_context(self,clause_text):
        # 1. 법령 검색
        laws = self.law_manager.search_relevent_laws(clause_text, k = 2)
        law_text = "\n".join(laws) if laws else "관련 법령 없음"

        # 2. 판례 검색
        precedents = search_precedents(clause_text, k = 1)
        precedent_text = precedents[0] if precedents else "관련 판례 없음"

        return f"[관련 법령]\n{law_text}\n\n[관련 판례]\n{precedent_text}"
    
    # DeepEval 라이브러리를 써서 독소조항 여부를 판별하는 함수
    def detect(self,clause_text):
        is_toxic = True
        risk_score = 0.6
        reason = ""
        retrieved_context = "retrieved context"
        return {
            "clause":clause_text,
            "is_toxic":is_toxic,
            "reason":reason,
            "context_used":retrieved_context
        }
    
    # 
    def generate_easy_suggestion(self,detection_result):
        prompt = ""
        return self.evaluator_llm.generate(prompt)

