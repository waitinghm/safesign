""" 판례 데이터 수집을 "joonhok-exo-ai/korean_law_open_data_precedents" 데이터셋에서 수행하도록 변경.
 쓰이지 않는 기존 API 코드는 주석 처리로 보관"""

# import os
# import requests
# import json
# import xml.etree.ElementTree as ET 
# from dotenv import load_dotenv

# # 환경 설정
# load_dotenv()
# MOLEG_API_KEY = os.getenv("MOLEG_API_KEY") 

# def search_precedent_list(query, display, page): 
#     """
#     [목록 검색] 법제처 판례 목록 API를 호출하여 판례 리스트와 총 개수를 반환합니다. (JSON 응답 파싱)
    
#     :param query: 검색할 키워드
#     :param display: 페이지당 검색할 판례의 개수
#     :param page: 검색할 페이지 번호
#     :return: (판례 목록 리스트, 총 검색 개수)
#     """
#     url = (
#         f"http://www.law.go.kr/DRF/lawSearch.do?OC={MOLEG_API_KEY}&target=prec"
#         f"&type=JSON&query={query}&display={display}&page={page}"
#     )
    
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         data = response.json()
        
#         prec_search = data.get("PrecSearch", {})
#         precedents = prec_search.get("prec", [])
#         total_count = int(prec_search.get("totalCnt", 0))
        
#         return precedents, total_count
#     except requests.exceptions.RequestException as e:
#         print(f"⚠️ 판례 목록 요청 실패 (쿼리: {query}, 페이지: {page}): {e}")
#         return [], 0
#     except json.JSONDecodeError:
#         print(f"⚠️ 판례 목록 JSON 파싱 실패 (쿼리: {query}, 페이지: {page}).")
#         return [], 0
#     except Exception as e:
#         print(f"⚠️ 판례 목록 검색 중 일반 오류: {e}")
#         return [], 0

# def get_precedent_detail_text(prec_id):
#     """
#     [상세 검색] 판례일련번호(prec_id)를 사용하여 상세 판결 내용(판결요지, 판시사항)을 조회하고 반환합니다.
    
#     :param prec_id: 판례일련번호
#     :return: (판결요지 리스트, 판시사항 텍스트)
#     """
#     if not prec_id: return None, None

#     # 상세 요청 URL: type=XML로 설정하여 판례 상세 전문을 요청
#     url = (
#         f"http://www.law.go.kr/DRF/lawService.do?OC={MOLEG_API_KEY}&target=prec"
#         f"&ID={prec_id}&type=XML" 
#     )

#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         root = ET.fromstring(response.content)
        
#         # 1. 판시사항 추출 및 HTML <br/> 태그 처리
#         holding_elem = root.find('판시사항')
#         holding = holding_elem.text.strip().replace('<br/>', '\n') if holding_elem is not None and holding_elem.text else ""
        
#         # 2. 판결요지 추출 (요지, 요지1, 요지2... 형태를 모두 포함하여 리스트로 정리)
#         summary_list = []
        
#         # <판결요지> 태그의 내용 추출
#         summary_elem = root.find('판결요지')
#         if summary_elem is not None and summary_elem.text:
#             summary_list.append(summary_elem.text.strip())
        
#         # 혹시 모를 요지N 형태의 태그 추출
#         i = 1
#         while True:
#             yoji_elem = root.find(f'요지{i}')
#             if yoji_elem is not None and yoji_elem.text:
#                 summary_list.append(yoji_elem.text.strip())
#                 i += 1
#             else:
#                 break
        
#         # 최종 리스트 정리: <br/> 태그를 개행 문자로 치환 및 공백 정리
#         summary_list = [s.replace('<br/>', '\n').strip() for s in summary_list if s.strip()]
        
#         if not summary_list and not holding:
#             return None, None

#         return summary_list, holding

#     except requests.exceptions.RequestException as e:
#         print(f"⚠️ 판례 상세 요청 실패 (ID:{prec_id}): {e}")
#         return None, None
#     except ET.ParseError:
#         print(f"⚠️ 판례 상세 XML 파싱 실패 (ID:{prec_id}).")
#         return None, None
#     except Exception as e:
#         print(f"⚠️ 판례 상세 검색 중 일반 오류: {e}")
#         return None, None

# def parse_precedent_content(summary_list, holding, prec_info):
#     """
#     추출된 판례 내용(요지, 판시사항)과 목록 정보를 통합하여 
#     RAG(Retrieval-Augmented Generation)에 사용될 단일 텍스트와 메타데이터를 생성합니다.

#     :param summary_list: 판결요지 리스트
#     :param holding: 판시사항 텍스트
#     :param prec_info: 판례 목록에서 얻은 기본 정보 (사건명, 사건번호 등)
#     :return: (임베딩될 통합 텍스트, 메타데이터 딕셔너리)
#     """
#     if not summary_list and not holding:
#         return None, None

#     # RAG에 사용할 텍스트 생성: 사건명, 사건번호, 판시사항, 판결요지 순으로 결합
#     content_raw = []
    
#     case_name = prec_info.get("사건명", "N/A")
#     case_number = prec_info.get("사건번호", "N/A")
    
#     content_raw.append(f"사건명: {case_name}")
#     content_raw.append(f"사건번호: {case_number}")
#     content_raw.append("-" * 30)

#     if holding:
#         content_raw.append("판시사항:")
#         content_raw.append(holding.strip())

#     if summary_list:
#         content_raw.append("\n판결요지:")
#         for i, summary in enumerate(summary_list):
#             content_raw.append(f"[{i+1}] {summary.strip()}")

#     full_text = "\n".join(content_raw).strip()
    
#     # 메타데이터 추출
#     metadata = {
#         "source_type": "판례",
#         "title": case_name,
#         "사건번호": case_number,
#         "판례일련번호": prec_info.get("판례일련번호", "N/A"),
#         "선고일자": prec_info.get("선고일자", "N/A"),
#         "법원명": prec_info.get("법원명", "N/A")
#     }

#     return full_text, metadata