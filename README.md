# safesign

폴더구조(수정가능)
```
Labor-Contract-Validator/
│
│
├── data/                      # 데이터 저장소 [cite: 312]
│   ├── vector_store/          # ChromaDB/FAISS (판례 임베딩 저장소)
│   └── raw_laws/              # 법제처 API 캐싱 데이터 (JSON)
│
├── src/                       # 핵심 소스 코드 (Backend & Logic)
│   ├── __init__.py
│   │
│   ├── parser/                # 계약서 파싱 및 청킹
│   │   ├── pdf_parser.py      # PyMuPDF/OCR 텍스트 추출
│   │   └── text_chunker.py    # 정규표현식 기반 조항 단위 Chunking 로직
│   │
│   ├── retriever/             # 법률/판례 검색기
│   │   ├── law_api.py         # 법제처 API 연동 모듈
│   │   └── case_search.py     # HuggingFace Vector Search 모듈
│   │
│   ├── evaluator/             # DeepEval 평가 로직 (핵심)
│   │   ├── g_eval.py          # 독소조항 판별용 G-Eval Metric 정의
│   │   ├── faithfulness.py    # 해석 신뢰성 검증 Metric 정의
│   │   └── test_cases.py      # 평가용 골든 데이터셋 (선택사항)
│   │
│   └── generator/             # 생성 및 리포팅
│       ├── llm_client.py      # Gemini/OpenAI 클라이언트 래퍼
│       └── report_gen.py      # 평가 결과 기반 '쉬운 해석' 생성 체인
│
├── ui/                        # 프론트엔드 컴포넌트
│   ├── dashboard.py           # 결과 시각화 (차트, 게이지)
│   ├── uploader.py            # 파일 업로드 위젯
│   └── layout.py              # 전체 페이지 레이아웃 관리
│
├── tests/                     # 단위 테스트 및 평가 실행
│   ├── test_parser.py
│   └── eval_experiment.py     # DeepEval 실험 실행 스크립트
│
├── .env                       # API KEY 관리 (Git 업로드 금지)
├── .gitignore                 # Git 무시 설정
├── LICENSE                    # MIT License
├── main.py                    # Streamlit 실행 진입점 (Entry Point)
├── README.md                  # 프로젝트 설명서
└── requirements.txt           # 의존성 라이브러리 목록
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.