from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

import torch
from keras.models import load_model

from transformers import BertModel, BertTokenizer
import google.generativeai as genai

import yt_dlp as youtube_dl
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import ast

from crawl_keyword import crawl_urls, crawl_news

import pytz
from datetime import timedelta
import datetime

import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 
app = Flask(__name__)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기본적 모델 불러오기

# KoBERT 모델과 토크나이저 로드
model_name = 'kykim/bert-kor-base'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.to(device)  # 모델을 GPU로 이동

# Gemini
genai.configure(api_key= os.getenv("API_KEY"))

safety_settings = [
    {
        "category" : "HARM_CATEGORY_HARASSMENT",
        "threshold" : "BLOCK_NONE"
    },
    {
        "category" : "HARM_CATEGORY_HATE_SPEECH",
        "threshold" : "BLOCK_NONE"
    },
    {
        "category" : "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold" : "BLOCK_NONE"
    },
    {
        "category" : "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold" : "BLOCK_NONE"
    },
]

gemini_model = genai.GenerativeModel('gemini-pro',safety_settings = safety_settings)

# 신뢰도 판단 모델 가중치
acc_model = load_model('./model.h5')

# youtube url에서 제목/업로드일자/자막 추출
def get_youtube_title(youtube_url):
    try:
        # 유튜브 영상 정보 가져오기
        with youtube_dl.YoutubeDL() as ydl:
            info = ydl.extract_info(youtube_url, download=False)

        # 유튜브 영상 제목 반환
        return info.get('title', '제목을 찾을 수 없습니다.')
    except Exception as e:
        return print("get_youtube_title_An error occurred:", e)

# youtube url에서 업로드 일자 추출
def get_upload_date(youtube_url):
    try:
        # 유튜브 URL 입력
        url = youtube_url

        # YouTube 객체 생성
        youtube = YouTube(url)

        # 업로드 일자 추출
        upload_date = youtube.publish_date
        # 한국 시간으로 변환 (옵션)
        KST = datetime.timezone(datetime.timedelta(hours=9))
        upload_date_kst = upload_date.astimezone(KST)
        
        return upload_date_kst
    except Exception as e:
        return print("get_upload_date_An error occurred:", e)  

# youtube url로 id 추출, 대본 추출
def extract_video_id(url):
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        # Extract query parameters
        query_params = parse_qs(parsed_url.query)
        # Get the video ID from query parameters (the 'v' parameter)
        video_id = query_params.get("v")

        if video_id:
            return video_id[0]  # 'v' parameter is a list, so we take the first item
        else:
            return "No video ID found in URL"
        
    except Exception as e:
        return print("extract_video_id_An error occurred:", e)


def get_transcript(video_id):
    try:
        # Fetch the auto-generated Korean transcript
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        script = ""
        for transcript in transcripts:
            script += transcript['text']
        return script
    except Exception as e:
        print("get_transcript_An error occurred:", e)  

def get_keyword(title, script):
    try:
        response = gemini_model.generate_content(title + "이 제목을 가진 유튜브 본문이 다음과 같아" + script + '이 제목 및 본문 내용의 "누가, 무엇을을 포함하는 핵심적인 중요한!! 내용을" 함축하면서도 연관성이 높은, 관련된 뉴스 기사를 찾기 좋은 "1~2단어"로 된 검색어를 [ {"keyword" : ... } ] 형식으로 1개만 추출해줘')
        print(response.text)
        return response.text
    except Exception as e:
        print("get_keyword_An error occurred:", e)

def filter_current(news_titles, news_contents, date_dict):
    
    news_titles = news_titles.copy()
    news_contents = news_contents.copy()
    date_dict = date_dict.copy()
    
    # 현재 날짜와 시간을 가져옵니다.
    current_datetime = datetime.datetime.now(pytz.timezone('Asia/Seoul'))

    # upload_date_kst를 현재 날짜로 설정합니다.
    upload_date_kst = current_datetime
    
    keys_list = list(date_dict.keys())

    date_news_titles = []
    date_news_contents = []
    term = 1

    while len(date_news_titles) < 8:
        for key, date in date_dict.items():
            news_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.timezone('Asia/Seoul'))
            one_day_after = upload_date_kst + timedelta(days=(-2) * term)
            two_days_after = upload_date_kst + timedelta(days=3 * term)

            if one_day_after <= news_date <= two_days_after:
                index = keys_list.index(key)
                if news_titles[index] not in date_news_titles:
                    date_news_titles.append(news_titles[index])
                    date_news_contents.append(news_contents[index])
                    print(f"Key: {key}, Index: {index}")

        term += 1

    return date_news_titles, date_news_contents

def filter_related(news_titles, news_contents, date_dict, upload_date_kst):
    
    news_titles = news_titles.copy()
    news_contents = news_contents.copy()
    date_dict = date_dict.copy()

    keys_list = list(date_dict.keys())

    date_news_titles = []
    date_news_contents = []
    term = 1

    while len(date_news_titles) < 8:
        for key, date in date_dict.items():
            news_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.timezone('Asia/Seoul'))
            one_day_after = upload_date_kst + timedelta(days=(-2) * term)
            two_days_after = upload_date_kst + timedelta(days=3 * term)

            if one_day_after <= news_date <= two_days_after:
                index = keys_list.index(key)
                if news_titles[index] not in date_news_titles:
                    date_news_titles.append(news_titles[index])
                    date_news_contents.append(news_contents[index])
                    print(f"Key: {key}, Index: {index}")

        term += 1  


# 한글이 아닌 문자 제거 함수
def clean_text(text):
    text = re.sub('[^가-힣\s]', '', text)
    return text

# 형태소 분석기 초기화
okt = Okt()

# 텍스트 전처리 함수
def preprocess_text(text):
    text = clean_text(text)  # 텍스트 정제
    tokenized = okt.morphs(text)  # 형태소 분석
    return ' '.join(tokenized)

# 불용어 제거 (stopwords 리스트를 만들어 사용)
stopwords = [
    '아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가',
    '으로', '로', '에게', '뿐이다', '의거하여', '근거하여', '입각하여', '기준으로', '예하면', '예를 들면', '예를 들자면',
    '저', '소인', '소생', '저희', '지말고', '하지마', '하지마라', '다른', '물론', '또한', '그리고', '비길수 없다', '해봐요',
    '습니까', '말하자면', '인 듯하다', '하지 않는다면', '만약에', '무엇', '무슨', '어느', '어떤', '아래', '위', '및', '만',
    '향하여', '무엇때문에', '그', '그런데', '그래서', '그러나', '그리고', '혹은', '또는', '바꾸어서', '말하면', '시키다',
    '하게 하다', '할 수 있다', '할 수 있어', '하는 편이 낫다', '불문하고', '했어요', '말할것도 없고', '무릎쓰고', '개의치않고',
    '하는것만 못하다', '하는것이 낫다', '매', '매번', '들', '모', '어느것', '어느', '로써', '갖고말하자면', '어디', '어느쪽',
    '어느것', '어느 해', '년', '월', '일', '령', '영', '영', '일', '이', '일', '고', '있다', '가', '이다', '하는', '그런', '그러한',
    '그럼', '그러므로', '그래', '그러니', '그러니까', '그러면', '네', '예', '우선', '누구', '누가', '알겠습니까', '요', '내', '내가',
    '말하자면', '또한', '그리고', '바꾸어서', '말하면', '나', '우리', '만일', '위에서', '서술한바와같이', '인', '등', '등등', '제',
    '신', '경우', '하고', '있다', '한', '이', '것', '들', '의', '있', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니',
    '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '말하', '알',
    '그러나', '받', '못하', '일', '그런', '또', '문제', '더', '사회', '많', '그리고', '좋', '크', '따르', '중', '나오', '가지', '씨',
    '시키', '만들', '지금', '생각하', '그러', '속', '하나', '집', '살', '모르', '적', '월', '데', '자신', '안', '어떤', '내', '내',
    '경우', '명', '생각', '시간', '그녀', '다시', '이런', '앞', '보이', '번', '나', '다른', '어떻', '여자', '개', '전', '들', '사실',
    '이렇', '점', '싶', '말', '정도', '좀', '원', '잘', '통하', '소리', '놓', '은', '는', '이', '가',
    '과', '와', '을', '를', '의', '에', '로', '를', '가', '이', '은', '는', '아니라', '에게서', '부터', '까지', '뿐만',
    '아니라', '밖에', '놓고', '아직', '로서', '해', '도', '에서는', '처럼', '대로', '까지는', '에는', '로는', '인데', '에서도',
    '에서만', '하려고', '하면', '만으로도', '따라서', '그래서', '그렇게', '그렇지만', '하지만', '그러므로', '그러니까', '그러면서',
    '그런데', '그리고', '또는', '혹은', '이라고', '라고']

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# 임베딩을 생성하는 함수
def get_bert_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        # inputs.to(device) 라인 제거 - GPU 사용하지 않음

        outputs = bert_model(**inputs)

        # 나머지 부분은 동일하게 유지
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        count = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / count

        return mean_pooled[0].detach().cpu().numpy()  # 결과를 CPU로 이동 (이미 CPU에서 실행 중이므로 필요 없으나 명시적으로 남겨둠)
    except Exception as e:
        print("get_bert_embedding: An error occurred:", e)

    
def get_top_five(title_acc, keyword):
    
    title_acc = title_acc.copy()
    
    # 딕셔너리를 value 값에 따라 정렬
    sorted_title_acc = dict(sorted(title_acc.items(), key=lambda item: item[1], reverse = True))
    keyword_embedding = get_bert_embedding(keyword).reshape(1, -1)

    # α 값을 설정 (예: 0.5)
    alpha = 0.8

    # 최종 점수를 계산하고 저장
    combined_scores = {}
    for title in sorted_title_acc.keys():
        title_embedding = get_bert_embedding(title).reshape(1, -1)
        similarity = cosine_similarity(keyword_embedding, title_embedding)[0][0]

        # 정규화된 정확도와 유사도를 결합
        combined_score = alpha * similarity + (1 - alpha) * title_acc[title]
        combined_scores[title] = combined_score

    # 결합된 점수에 따라 정렬
    sorted_combined_scores = dict(sorted(combined_scores.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_combined_scores

@app.route('/youtubeNews/related', methods=['POST'])
def youtubeNewsRelated():
    if request.method == 'POST':
        try: 
            data = request.json  # 클라이언트에서 보낸 데이터를 JSON 형태로 받음
            youtube_url = data['url'] 
        
            
            title = get_youtube_title(youtube_url)
            upload_date_kst = get_upload_date(youtube_url)
            video_id = extract_video_id(youtube_url)
            script = get_transcript(video_id)
            keyword_json = get_keyword(title, script)
            
            data_dict = ast.literal_eval(keyword_json)
            if type(data_dict) == dict:
                keyword = data_dict["keyword"]
            elif type(data_dict) == list:
                keyword = data_dict[0]["keyword"]
                
            print("★제목, 업로드 날짜, 대본, 키워드 추출 완료\n")
                
            final_urls = crawl_urls(keyword)
            print("★url 목록들 받아오기 완료\n")
            
            news_titles, news_contents, news_dates = crawl_news(final_urls)  
            print("★크롤링 완료\n")
            
            date_dict = dict(zip(news_titles, news_dates)) 
            
            curr_date_news_titles, curr_date_news_contents = filter_current(news_titles, news_contents, date_dict) 
            print("★curr ... 날짜 기반 필터링 완료\n") 
            # rel_date_news_titles, rel_date_news_contents = filter_related(news_titles, news_contents, date_dict, upload_date_kst)
            # print("★rel ... 날짜 기반 필터링 완료\n")    
            
            curr_result_dict = dict(zip(curr_date_news_titles, curr_date_news_contents))
            # rel_result_dict = dict(zip(rel_date_news_titles, rel_date_news_contents))
            
            curr_title_acc = {}
            curr_title_embedding = {}
            # rel_title_acc = {}
            # rel_title_embedding = {}

            for title in curr_result_dict.keys():
                curr_input_text = remove_stopwords(preprocess_text(title))
                curr_input_embedding = get_bert_embedding(curr_input_text)

                # 입력 데이터가 올바른 형태인지 확인하고 필요한 경우 형태 변환
                if len(curr_input_embedding.shape) == 1:
                    curr_input_embedding = np.reshape(curr_input_embedding, (1, -1))

                # 모델을 사용한 예측 수행
                curr_predictions = acc_model.predict(curr_input_embedding)

                curr_title_acc[title] = curr_predictions
                curr_title_embedding[title] = curr_input_embedding
            print("★curr ... 신뢰도 예측 완료\n")
            
            # for title in rel_result_dict.keys():
            #     rel_input_text = remove_stopwords(preprocess_text(title))
            #     rel_input_embedding = get_bert_embedding(rel_input_text)

            #     # 입력 데이터가 올바른 형태인지 확인하고 필요한 경우 형태 변환
            #     if len(rel_input_embedding.shape) == 1:
            #         rel_input_embedding = np.reshape(rel_input_embedding, (1, -1))

            #     # 모델을 사용한 예측 수행
            #     rel_predictions = acc_model.predict(rel_input_embedding)

            #     rel_title_acc[title] = rel_predictions
            #     rel_title_embedding[title] = rel_input_embedding
            # print("★rel ... 신뢰도 예측 완료\n")
            
            curr_sorted_combined_scores = get_top_five(curr_title_acc, keyword)
            print("★curr ... top5 완료\n")
            # rel_sorted_combined_scores = get_top_five(rel_title_acc, keyword)
            # print("★rel ... top5 완료\n")
            
            # 결과 출력
            curr_top_5_combined = [{ "title" : key.strip(), "article" : curr_result_dict[key].strip() } for key in list(curr_sorted_combined_scores)[:5]]
            # rel_top_5_combined = [{ "title" : key.strip(), "article" : curr_result_dict[key].strip() } for key in list(rel_sorted_combined_scores)[:5]]
            
            result = {"curr_youtube_news" : curr_top_5_combined,
                    # "rel_youtube_news" : rel_top_5_combined
                    }  # 예시 결과
            
            return jsonify(result)

        except Exception as e:
            print("An error occurred:", e)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


