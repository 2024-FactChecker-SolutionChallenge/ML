from bs4 import BeautifulSoup
import requests
import re
import datetime
from tqdm import tqdm
import sys

def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num+1
    else:
        return num+9*(num-1)


def makeUrl(search, start_pg, end_pg):
    if start_pg == end_pg:
        urls = []
        start_page = makePgNum(start_pg)
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(start_page)
        urls.append(url)
        print("생성url: ", url)
        return urls
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page)
            urls.append(url)
        return urls

def news_attrs_crawler(articles,attrs):
    attrs_content=[]
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

def articles_crawler(url):
    original_html = requests.get(url,headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver,'href')
    return url

def crawl_urls(keyword):
    search = keyword
    page = 1 
    print("\n크롤링할 시작 페이지: ",page,"페이지")
    page2 = 10 
    print("\n크롤링할 종료 페이지: ",page2,"페이지")


    url = makeUrl(search,page,page2)

    news_url =[]
    
    for url_item in url:
        article_urls = articles_crawler(url_item)
        news_url.append(article_urls)



    def makeList(newlist, content):
        for sublist in content:
            newlist.extend(sublist)
        return newlist



    news_url_1 = []
    makeList(news_url_1,news_url)
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass
    
    return final_urls

def crawl_news(final_urls):
    
    news_titles = []
    news_contents =[]
    news_dates = []
    final_urls = final_urls.copy()
    
    for i in tqdm(final_urls):
        news = requests.get(i,headers=headers)
        news_html = BeautifulSoup(news.text,"html.parser")
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")

        content = news_html.select("article#dic_area")
        if content == []:
            content = news_html.select("#articeBody")

        content = ''.join(str(content))

        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')

        news_titles.append(title)
        news_contents.append(content)

        try:
            html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
            news_date = re.sub(pattern=pattern1,repl='',string=str(news_date))
            news_dates.append(news_date)
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")

    return news_titles, news_contents, news_dates
    
    