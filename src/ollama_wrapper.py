import ollama
from deepeval.models.base_model import DeepEvalBaseLLM

class OllamaDeepEvalWrapper(DeepEvalBaseLLM):
    """
    DeepEval에서 Ollama 모델을 사용하기 위한 래퍼 클래스
    (LangChain을 사용하지 않고 공식 ollama 라이브러리를 직접 호출하여 호환성 확보)
    """
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        # ollama 라이브러리는 별도 클라이언트 객체 생성 불필요

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        """
        공식 ollama.chat 함수를 사용하여 답변을 생성합니다.
        복잡한 모델명(hf.co/...)도 문제없이 처리합니다.
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
                stream=False # DeepEval 평가용이므로 스트림 없이 전체 응답을 한 번에 받음
            )
            return response['message']['content']
        except Exception as e:
            return f"Ollama Generation Error: {e}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name