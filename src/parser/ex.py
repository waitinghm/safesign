import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import re

# ==============================================================================
# 1. 환경 설정 및 API 정보
# ==============================================================================
BASE_URL = 'http://www.law.go.kr'
SEARCH_URL = f'{BASE_URL}/DRF/lawSearch.do'

# API 요청 기본 파라미터 (단일 항목 추출을 위한 설정은 변경 없음)
BASE_PARAMS = {
    "OC": "junhajs", 
    "target": "moelCgmExpc",  # 고용노동부 질의회시/행정해석
    "type": "JSON",           
    "query": "퇴직",           
    "numOfRows": 1,           # 1개만 요청하여 첫 항목만 가져옵니다.
    "page": 1                 
}

# 최종 데이터 저장 리스트
all_extracted_data = []


# ==============================================================================
# 2. 핵심 함수 (이전 코드와 동일)
# ==============================================================================

def extract_final_detail_url(wrapper_html: str) -> str or None:
    """1차 요청으로 받은 HTML Wrapper에서 실제 내용이 담긴 2차 URL을 추출합니다."""
    try:
        soup = BeautifulSoup(wrapper_html, 'html.parser')
        
        # id가 'url'인 input 태그의 value 속성에서 URL 추출
        url_input = soup.find('input', {'id': 'url'})
        if url_input and url_input.get('value'):
            return url_input.get('value')
        
        # <iframe> 태그의 src 속성에서 추출 (예비)
        iframe = soup.find('iframe')
        if iframe and iframe.get('src'):
            return iframe.get('src')
            
        return None
    except Exception:
        return None


def clean_detail_content(detail_html: str) -> str:
    """
    최종 상세 페이지 HTML에서 핵심 해석 텍스트를 정확히 추출하고 클리닝합니다.
    (노이즈의 시작점/끝점을 활용한 정밀 필터링)
    """
    soup = BeautifulSoup(detail_html, 'html.parser')
    
    # HTML body 전체 텍스트를 추출 (가장 넓은 범위에서 시작)
    main_content = soup.body
    if not main_content:
        return "콘텐츠 추출 실패"

    text_content = main_content.get_text(separator='\n', strip=True)
    
    # 1. 핵심 내용 시작점 찾기: '【질의요지】' 이전 내용 제거
    start_marker = "【질의요지】"
    if start_marker in text_content:
        text_content = text_content[text_content.find(start_marker):]

    # 2. 핵심 내용 끝점 찾기: 불필요한 Footer/안내 문구 직전에서 내용 자르기
    end_marker_1 = "【중앙부처 1차 해석에 대한 안내】"
    end_marker_2 = "검색조문선택" # 파일 다운로드/검색 관련 노이즈 시작점
    
    end_index = len(text_content)
    
    # 두 종료 마커 중 먼저 나타나는 지점 찾기
    if end_marker_1 in text_content:
        end_index = min(end_index, text_content.find(end_marker_1))
    
    if end_marker_2 in text_content:
        end_index = min(end_index, text_content.find(end_marker_2))
        
    # 핵심 내용만 남기기 위해 잘라내기
    text_content = text_content[:end_index].strip()


    # 3. 추가적인 줄 단위 정리 (제목, 전화번호 등 잔여 노이즈 제거)
    final_lines = []
    
    # 제거할 잔여 노이즈 패턴 정의
    line_noise_patterns = [
        "본문 바로가기",
        "고용노동부 누리집",
        "에서 수집한 데이터입니다.",
        r"\d{3}-\d{3}-\d{4}" # 전화번호 패턴
    ]

    for line in text_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 잔여 노이즈 패턴이 포함된 줄 제거
        is_noise = False
        for pattern in line_noise_patterns:
            if re.search(pattern, line):
                is_noise = True
                break
        
        # '고용노동부(퇴직연금복지과)' 같은 기관명 줄 제거 (이미 제목에 있으므로)
        if "고용노동부(" in line and "퇴직연금복지과" in line and len(line) < 50:
             is_noise = True
             
        if not is_noise:
            final_lines.append(line)

    return '\n'.join(final_lines)


def fetch_and_clean_detail_content(detail_url: str) -> str:
    """1차 URL로 요청하여 2차 URL을 추출하고, 2차 URL로 다시 요청하여 최종 내용을 클리닝합니다."""
    try:
        # 1차 요청 (Wrapper HTML)
        wrapper_response = requests.get(detail_url, timeout=30)
        wrapper_response.raise_for_status() 
        wrapper_html = wrapper_response.text
        
        # 2차 URL 추출
        final_url = extract_final_detail_url(wrapper_html)
        if not final_url:
            return "최종 콘텐츠 URL 추출 실패"

        # 2차 요청 (Final Content HTML)
        final_response = requests.get(final_url, timeout=30)
        final_response.raise_for_status() 
        final_html = final_response.text
        
        # 최종 텍스트 클리닝
        content_raw = clean_detail_content(final_html)
        return content_raw

    except requests.exceptions.RequestException as e:
        return f"[Error] 요청 실패: {e}"
    except Exception as e:
        return f"[Error] 알 수 없는 오류: {e}"


# ==============================================================================
# 3. 메인 로직: 단일 항목 데이터 수집 및 출력
# ==============================================================================

def fetch_single_item():
    """
    단일 항목을 수집하고 그 결과를 출력합니다.
    """
    print(f"[키워드: {BASE_PARAMS['query']}] 첫 페이지, 첫 항목 데이터 수집 시작...")
    
    # 1. 초기 요청 및 단일 항목 파악
    try:
        initial_response = requests.get(SEARCH_URL, params=BASE_PARAMS, timeout=30)
        initial_response.raise_for_status()
        initial_data = initial_response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 실패: {e}")
        return
    except json.JSONDecodeError:
        print("API 응답이 유효한 JSON 형식이 아닙니다.")
        return

    items = initial_data.get('CgmExpc', {}).get('cgmExpc', [])
    
    if not items:
        print("검색 결과가 없습니다.")
        return

    # 2. 첫 번째 항목만 처리
    item = items[0]
    source_id = item.get("법령해석일련번호")
    title = item.get("안건명")
    detail_link = item.get("법령해석상세링크")
    
    # 절대 경로 생성
    full_detail_url = f"{BASE_URL}{detail_link}"
    
    print(f"\n--- ID {source_id} ({title}) 상세 내용 다운로드 시도 ---")

    # 3. 상세 내용 요청 및 클리닝
    content_raw = fetch_and_clean_detail_content(full_detail_url)

    # 4. 최종 데이터 구조 출력
    extracted = {
        "source_id": source_id,
        "source_type": "행정해석",
        "title": title,
        "issue_date": item.get("해석일자"),
        "source_agency": item.get("해석기관명"),
        "content_raw": content_raw,
        "detail_url": full_detail_url
    }
    
    print(f"\n=======================================================")
    print(f"✅ 단일 항목 수집 결과 ({source_id})")
    print("=======================================================")
    for key, value in extracted.items():
        if key == 'content_raw':
            print(f"**{key}:**\n{value}") 
        else:
            print(f"**{key}:** {value}")
    print("=======================================================")


if __name__ == "__main__":
    fetch_single_item()