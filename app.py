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

from crawl_home import re_tag, art_crawl
from tqdm import tqdm

import time

load_dotenv() 
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'kykim/bert-kor-base'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.to(device) 

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

acc_model = load_model('./model.h5')

def get_youtube_title(youtube_url):
    try:
        with youtube_dl.YoutubeDL() as ydl:
            info = ydl.extract_info(youtube_url, download=False)

        return info.get('title', '제목을 찾을 수 없습니다.')
    except Exception as e:
        return print("get_youtube_title_An error occurred:", e)

def get_upload_date(youtube_url):
    try:
        url = youtube_url

        youtube = YouTube(url)

        upload_date = youtube.publish_date
        KST = datetime.timezone(datetime.timedelta(hours=9))
        upload_date_kst = upload_date.astimezone(KST)
        
        return upload_date_kst
    except Exception as e:
        return print("get_upload_date_An error occurred:", e)  

def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v")

        if video_id:
            return video_id[0]  
        else:
            return "No video ID found in URL"
        
    except Exception as e:
        return print("extract_video_id_An error occurred:", e)

def get_transcript(video_id):
    try:
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        script = ""
        for transcript in transcripts:
            script += transcript['text']
        return script
    except Exception as e:
        print("get_transcript_An error occurred:", e)  

def get_keyword(title, script):
    try:
        response = gemini_model.generate_content(title + "이 제목을 가진 영상 대본 본문이 다음과 같아" + script + '이 제목 및 본문 내용의 요지를 함축하면서도 연관성이 높은, 이 영상의 사실 여부를 따지기 위한 관련 뉴스 기사를 찾기 좋은 "1~3단어!!!!!!!"로 된 검색어를 [ {"keyword" : ... } ] 형식으로 1개만 추출해줘')
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
            if len(date_news_titles) > 20:
                return date_news_titles, date_news_contents
            
            news_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.timezone('Asia/Seoul'))
            one_day_after = upload_date_kst + timedelta(days=(-2) * term)
            two_days_after = upload_date_kst + timedelta(days=3 * term)

            if one_day_after <= news_date <= two_days_after:
                index = keys_list.index(key)
                if news_titles[index] not in date_news_titles:
                    date_news_titles.append(news_titles[index])
                    date_news_contents.append(news_contents[index])

        term += 1

    return date_news_titles, date_news_contents

def filter_related(news_titles, news_contents, date_dict, upload_date_kst):
    
    print("유튜브 업로드 날짜 : ", upload_date_kst)
    
    news_titles = news_titles.copy()
    news_contents = news_contents.copy()
    date_dict = date_dict.copy()

    keys_list = list(date_dict.keys())

    date_news_titles = []
    date_news_contents = []
    term = 1

    while len(date_news_titles) < 8:
        for key, date in date_dict.items():
            if len(date_news_titles) > 20:
                return date_news_titles, date_news_contents
            
            news_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.timezone('Asia/Seoul'))
            one_day_after = upload_date_kst + timedelta(days=(-2) * term)
            two_days_after = upload_date_kst + timedelta(days=3 * term)

            if one_day_after <= news_date <= two_days_after:
                index = keys_list.index(key)
                if news_titles[index] not in date_news_titles:
                    date_news_titles.append(news_titles[index])
                    date_news_contents.append(news_contents[index])

        term += 1  
    return date_news_titles, date_news_contents

def clean_text(text):
    text = re.sub('[^가-힣\s]', '', text)
    return text

okt = Okt()

def preprocess_text(text):
    text = clean_text(text)  
    tokenized = okt.morphs(text)  
    return ' '.join(tokenized)

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

def get_bert_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        outputs = bert_model(**inputs)

        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        count = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / count

        return mean_pooled[0].detach().cpu().numpy() 
    except Exception as e:
        print("get_bert_embedding: An error occurred:", e)

def get_batch_bert_embedding(texts):
    try:
        print(f"Processing {len(texts)} texts...") 

        batch = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        print("Tokenization completed, starting embeddings calculation...")

        with torch.no_grad():
            outputs = bert_model(**batch)

        embeddings = outputs.last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        count = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / count

        print("Embeddings calculation completed.")

        return mean_pooled.detach().cpu().numpy()
    except Exception as e:
        print("get_batch_bert_embedding: An error occurred:", e)

    
def get_top_five(title_acc, keyword):
    
    title_acc = title_acc.copy()
    
    sorted_title_acc = dict(sorted(title_acc.items(), key=lambda item: item[1], reverse = True))
    keyword_embedding = get_bert_embedding(keyword).reshape(1, -1)

    alpha = 0.8

    combined_scores = {}
    for title in sorted_title_acc.keys():
        title_embedding = get_bert_embedding(title).reshape(1, -1)
        similarity = cosine_similarity(keyword_embedding, title_embedding)[0][0]

        combined_score = alpha * similarity + (1 - alpha) * title_acc[title]
        combined_scores[title] = combined_score
        
    sorted_combined_scores = dict(sorted(combined_scores.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_combined_scores


def home_get_top_five(title_acc):
    
    title_acc = title_acc.copy()
    
    sorted_title_acc = dict(sorted(title_acc.items(), key=lambda item: item[1], reverse = True))

    sorted_scores = dict(sorted(sorted_title_acc.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_scores

@app.route('/youtubeNews/related', methods=['POST'])
def youtubeNewsRelated():
    if request.method == 'POST':
        try: 
            data = request.json  
            youtube_url = data['url'] 
            
            if 'youtu.be' in youtube_url:
                video_id = youtube_url.split('/')[-1].split('?')[0]

                params = ''
                if '?' in youtube_url and '&' in youtube_url.split('?')[1]:
                    params = '&' + youtube_url.split('?')[1]

                youtube_url = f"https://youtube.com/watch?v={video_id}{params}"
            
            yt_title = get_youtube_title(youtube_url)
            upload_date_kst = get_upload_date(youtube_url)
            video_id = extract_video_id(youtube_url)
            script = get_transcript(video_id)
            
            max_retries = 1  
            retry_delay = 2  
            for attempt in range(max_retries):
                try:
                    keyword_json = get_keyword(yt_title, script)
                    break 
                except Exception as e:
                    print(f"시도 {attempt + 1}/{max_retries}: 에러 발생 - {e}")
                    time.sleep(retry_delay)
            else:
                print(f"{max_retries}회 시도 후에도 성공하지 못했습니다.")
                keyword_json = None 
            
            data_dict = ast.literal_eval(keyword_json)
            if type(data_dict) == dict:
                keyword = data_dict["keyword"]
            elif type(data_dict) == list:
                keyword = data_dict[0]["keyword"]
                
            print("★제목, 업로드 날짜, 대본, 키워드 추출 완료\n")
            print(yt_title)
                
            final_urls = crawl_urls(keyword)
            print("★url 목록들 받아오기 완료\n")
            print(yt_title)
            
            news_titles, news_contents, news_dates = crawl_news(final_urls)  
            print("★크롤링 완료\n")
            print(yt_title)
            
            date_dict = dict(zip(news_titles, news_dates)) 
            
            curr_date_news_titles, curr_date_news_contents = filter_current(news_titles, news_contents, date_dict) 
            print("★curr ... 날짜 기반 필터링 완료\n") 
            rel_date_news_titles, rel_date_news_contents = filter_related(news_titles, news_contents, date_dict, upload_date_kst)
            print("★rel ... 날짜 기반 필터링 완료\n")   
            
            print(yt_title) 
            
            curr_result_dict = dict(zip(curr_date_news_titles, curr_date_news_contents))
            rel_result_dict = dict(zip(rel_date_news_titles, rel_date_news_contents))
            
            curr_title_acc = {}
            curr_title_embedding = {}
            rel_title_acc = {}
            rel_title_embedding = {}
                
            titles = list(curr_result_dict.keys())
            processed_titles = [remove_stopwords(preprocess_text(title)) for title in titles]
            batch_embeddings = get_batch_bert_embedding(processed_titles)
            
            predictions = acc_model.predict(batch_embeddings)

            for title, prediction, embedding in zip(titles, predictions, batch_embeddings):
                curr_title_acc[title] = prediction
                curr_title_embedding[title] = embedding
            print("★curr ... 신뢰도 예측 완료\n")
            
            titles = list(rel_result_dict.keys())
            processed_titles = [remove_stopwords(preprocess_text(title)) for title in titles]
            batch_embeddings = get_batch_bert_embedding(processed_titles)

            predictions = acc_model.predict(batch_embeddings)

            for title, prediction, embedding in zip(titles, predictions, batch_embeddings):
                rel_title_acc[title] = prediction
                rel_title_embedding[title] = embedding
            print("★rel ... 신뢰도 예측 완료\n")
            
            print(yt_title)
            
            curr_sorted_combined_scores = get_top_five(curr_title_acc, keyword)
            print("★curr ... top5 완료\n")
            rel_sorted_combined_scores = get_top_five(rel_title_acc, keyword)
            print("★rel ... top5 완료\n")
            
            print(yt_title)

            curr_top_5_combined = [{ "title": str(key).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip(), 
                                    "article": str(" ".join(map(str, curr_result_dict[key])) if isinstance(curr_result_dict[key], list) else str(curr_result_dict[key]).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip())[curr_result_dict[key].find('[') + 1: curr_result_dict[key].rfind(']')],
                                    "date" : str(date_dict[key]),
                                    "credibility" : float(curr_sorted_combined_scores[key])}
                                for key in list(curr_sorted_combined_scores)[:5]]
            print("★curr_top_5_combined ... top5 완료\n")

            rel_top_5_combined = [{ "title": str(key).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip(), 
                                    "article": str(" ".join(map(str, rel_result_dict[key])) if isinstance(rel_result_dict[key], list) else str(rel_result_dict[key]).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip())[rel_result_dict[key].find('[') + 1 : rel_result_dict[key].rfind(']')],
                                    "date" : str(date_dict[key]),
                                    "credibility" : float(rel_sorted_combined_scores[key])} 
                                for key in list(rel_sorted_combined_scores)[:5]]
            print("★rel_top_5_combined ... top5 완료\n")
            
            print(yt_title)

            result = { "yt_title" : str(yt_title), 
                    "upload_date": str(upload_date_kst),
                    "keyword" : keyword,
                    "curr_youtube_news": curr_top_5_combined,
                    "rel_youtube_news": rel_top_5_combined}  

            
            return jsonify(result)

        except Exception as e:
            print("An error occurred:", e)
            return jsonify({"error": str(e)}), 500
        
@app.route('/interests', methods=['GET'])
def interests():
    if request.method == 'GET':
        try: 
            all_hrefs = {}
            sids = [i for i in range(100,106)] 

            for sid in sids:
                sid_data = re_tag(sid)
                all_hrefs[sid] = sid_data
            
            section_lst = [s for s in range(100, 106)]
            artdic_lst = []

            for section in tqdm(section_lst, desc="Processing sections"):
                for i in range(len(all_hrefs[section][:20])):
                    if  "news.naver.com" in all_hrefs[section][i]:
                        art_dic = art_crawl(all_hrefs, section, i)
                        art_dic["section"] = section
                        art_dic["url"] = all_hrefs[section][i]
                        artdic_lst.append(art_dic)
                    
            print("★ home_크롤링 완료")
            
            title_main_dict = {item['title']: item['main'] for item in artdic_lst}
            title_date_dict = {item['title']: item['date'] for item in artdic_lst}
            title_section_dict = {item['title']: item['section'] for item in artdic_lst}

            home_title_acc = {}
            home_title_embedding = {}

            news_titles = [remove_stopwords(preprocess_text(news['title'])) for news in artdic_lst]

            batch_embeddings = get_batch_bert_embedding(news_titles)

            predictions = acc_model.predict(batch_embeddings)

            for i, news in enumerate(artdic_lst):
                home_title_acc[news['title']] = predictions[i]
                home_title_embedding[news['title']] = batch_embeddings[i]

            print("home_예측 완료")
            
            home_sorted_scores = home_get_top_five(home_title_acc)
            print("★ home_top5완료")

            result = [{ "title": str(key).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip(), 
                        "article": str(title_main_dict[key]).replace('"', ' ').replace('\n', ' ').replace('\t', ' ').strip(),
                        "date" : str(title_date_dict [key]),
                        "credibility" : float(home_sorted_scores[key]),
                        "section" : title_section_dict[key]}
                    for key in list(home_sorted_scores)]
            
            result = [news for news in result if news["article"] != ''] 
            
            return jsonify(result)
            
        except Exception as e:
            print("An error occurred:", e)
            return jsonify({"error": str(e)}), 500
        
@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        try: 
            data = request.json 
            article = data['article'] 
            summary = data['summary']
            
            response = gemini_model.generate_content(article + "이 본문을 가지고 요약을 1~3줄로 다음과 같이 해봤어 : " + summary + '이 기사 내용을 잘 요약한건지 3~4줄로 피드백을 해주고 더 나은 요약을 제공해줘')
            result = {"feedback" : response.text}
            return jsonify(result)
            
        except Exception as e:
            print("An error occurred:", e)
            return jsonify({"error": str(e)}), 500
        
@app.route('/study/daily-quiz/quiz-word', methods=['GET'])
def quiz_word():
    if request.method == 'GET':
        try:
            response = gemini_model.generate_content('다소 어려운, 요즘 시사 뉴스에서 흔히 볼 수 있는, 시사/경제 한글어휘들 20개를 [{"quiz_word" : "단어1", "mean" : "단어1에 대한 뜻풀이"}, {"quiz_word" : "단어2", "mean" : "단어2에 대한 뜻풀이"}, .... ] 와 같은 형태로 반환해줘. 뜻은 1줄 이내의 형식으로 반환해줘.')
            words_json = response.text
            print(words_json)
            
            data_dict = ast.literal_eval(words_json)
            print(data_dict)
            print(type(data_dict))
            
            if type(data_dict) == dict:
                result = [data_dict]
            elif type(data_dict) == list:
                result = data_dict
                
            for index, item in enumerate(result):
                item['id'] = index
            
            return jsonify({"words" : result})
            
        except Exception as e:
            print("An error occurred:", e)
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


