# 출처: https://bigdata-doctrine.tistory.com/34 [경제와 데이터:티스토리]

import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

def ex_tag(sid, page):
    ### 뉴스 분야(sid)와 페이지(page)를 입력하면 그에 대한 링크들을 리스트로 추출하는 함수 ###
    
    ## 1.
    url = f"https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={sid}"\
    "#&date=%2000:00:00&page={page}"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"\
    "(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "\
    "Chrome/110.0.0.0 Safari/537.36"})
    soup = BeautifulSoup(html.text, "lxml")
    a_tag = soup.find_all("a")
    
    ## 2.
    tag_lst = []
    
    # 모든 a 태그 찾기
    a_tags = soup.find_all('a')

    # 각 a 태그의 href 속성 출력
    for tag in a_tags:
        href = tag.get('href')
        if href:  # href 속성이 있는 경우에만 출력
            # print(href)
            if str(href).startswith("https://n.news.naver.com/"):
                tag_lst.append(href)
                
    return tag_lst

# 정치 : 100 / 경제 : 101 / 사회 : 102 / 세계 : 104 / IT과학 : 105 / 생활문화 : 103


def re_tag(sid):
    ### 특정 분야의 1페이지까지의 뉴스의 링크를 수집하여 중복 제거한 리스트로 변환하는 함수 ###
    re_lst = []
    for i in tqdm(range(1)):
        lst = ex_tag(sid, i+1)
        re_lst.extend(lst)

    # 중복 제거
    re_set = set(re_lst)
    re_lst = list(re_set)
    
    return re_lst

def art_crawl(all_hrefs, sid, index):
    
    all_hrefs = all_hrefs.copy()
    
    """
    sid와 링크 인덱스를 넣으면 기사제목, 날짜, 본문을 크롤링하여 딕셔너리를 출력하는 함수 
    
    Args: 
        all_hrefs(dict): 각 분야별로 100페이지까지 링크를 수집한 딕셔너리 (key: 분야(sid), value: 링크)
        sid(int): 분야 [100: 정치, 101: 경제, 102: 사회, 103: 생활/문화, 104: 세계, 105: IT/과학]
        index(int): 링크의 인덱스
    
    Returns:
        dict: 기사제목, 날짜, 본문이 크롤링된 딕셔너리
    
    """
    art_dic = {}
    
    ## 1.
    title_selector = "#title_area > span"
    date_selector = "#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans"\
    "> div.media_end_head_info_datestamp > div:nth-child(1) > span"
    main_selector = "#dic_area"
    
    url = all_hrefs[sid][index]
    html = requests.get(url, headers = {"User-Agent": "Mozilla/5.0 "\
    "(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"\
    "Chrome/110.0.0.0 Safari/537.36"})
    soup = BeautifulSoup(html.text, "lxml")
    
    ## 2.
    # 제목 수집
    title = soup.select(title_selector)
    title_lst = [t.text for t in title]
    title_str = "".join(title_lst)
    
    # 날짜 수집
    date = soup.select(date_selector)
    date_lst = [d.text for d in date]
    date_str = "".join(date_lst)
    
    # 본문 수집
    main = soup.select(main_selector)
    main_lst = []
    for m in main:
        m_text = m.text
        m_text = m_text.strip()
        main_lst.append(m_text)
    main_str = "".join(main_lst)
    
    ## 3.
    art_dic["title"] = title_str
    art_dic["date"] = date_str
    art_dic["main"] = main_str
    
    return art_dic
