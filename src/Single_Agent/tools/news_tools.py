import operator
from typing import List, Dict, Any, Optional, Literal, TypedDict
from langchain_core.tools import tool
import os, sys
import sqlite3, json
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(root_dir)

from src.Crawling.news_crawling import News_Crawler
from src.Crawling.news_db import News_Database

from selenium import webdriver
import warnings
warnings.filterwarnings('ignore')

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")

from src.utils.loading_utils import load_ticker_map
tickers = load_ticker_map()

@tool
def search_news_headlines(ticker : str, count = 5) -> list:
    """
    특정 기업의 최신 뉴스 헤드라인과 html 경로를 검색합니다. 
    Args:
        ticker : 기업 티커 (예: "NVDA", "BOA")
    Returns:
        뉴스 헤드라인과 html 경로를 담은 리스트 (JSON 형식의 문자열)
    """
    print(f"\n[Tool Call] '{ticker}' 관련 뉴스 헤드라인 검색 중...")
    news_crawler = News_Crawler()
    news_db = News_Database()

    company_name = tickers[ticker]
    db_path = f"data/News_DB/{ticker}.db"
    print(f"\n[Tool] {company_name}의 뉴스 {count}개 수집 및 DB 적재 시작...")
    _, html_paths = news_crawler.get_news_html_count(ticker, count, chrome_options)
    new_html_paths = []
    print("--- 뉴스 html 수집 완료 ---")
    print("--------------------------------")
    if os.path.exists(db_path): # 이미 존재하는 db라면, 즉 크롤링 결과가 이미 존재한다면
        new_html_paths = [path for path in html_paths if news_db.compare_news_db(db_path, path)]
        if len(new_html_paths) == 0:
            print("새로운 뉴스가 없습니다. (추가 수집 건너뜀)")
        else:
            print(f"새로운 뉴스 {len(new_html_paths)}개를 찾았습니다.")
    else:
        print("DB가 존재하지 않습니다. (새로운 DB 생성)")
        new_html_paths = html_paths

    # 새롭게 수집할 뉴스 존재
    if new_html_paths:
        scraped_news = news_crawler.get_news_content(new_html_paths, chrome_options)
        news_db.save_data_to_db(scraped_news, db_path)
        print("--- 뉴스 DB 저장 완료 ---")

    if not html_paths:
        return "현재 수집 가능한 뉴스가 없습니다."

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # html_paths에 있는 뉴스들만 가져옴
    html_placeholder = ",".join(["?"] * len(html_paths))
    query = f"""
        SELECT A.article_id, A.title, A.editor, A.date, A.html
        FROM Articles A
        WHERE A.html IN ({html_placeholder})
        ORDER BY A.date DESC
    """
    cursor.execute(query, html_paths)
    rows = cursor.fetchall()
    conn.close()

    news_articles = []
    for row in rows:
        news_article = {
            "id" : row["article_id"],
            "title" : row["title"],
            "date" : row["date"],
            "html" : row["html"]
        }
        news_articles.append(news_article)

    return json.dumps(news_articles, ensure_ascii = False)

@tool
def read_news_content(ticker : str, article_ids : List[int]) -> str: 
    """
    특정 뉴스 기사들의 html을 받아 상세 본문을 반환합니다.
    Args:
        ticker : 기업 티커 (예: "NVDA", "BOA")
        article_ids : 뉴스 기사들의 기사 ID 리스트 (예: [1, 3])
    """
    print(f"\n[Tool Call] '{ticker}' 기사 {article_ids} 뉴스 본문 읽기 중...")

    db_path = f"data/News_DB/{ticker}.db"
    if not os.path.exists(db_path):
        return "DB 파일이 없습니다. 먼저 뉴스를 검색해주세요."
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ID 리스트로 쿼리 생성
    ids_placeholder = ",".join(["?"] * len(article_ids))
    query = f"""
        SELECT A.article_id, A.title, A.editor, A.date, A.html,
               C.content, C.block_type
        FROM Articles A
        LEFT JOIN Content C ON A.article_id = C.article_id
        WHERE A.article_id IN ({ids_placeholder})
        ORDER BY A.date DESC
    """

    try:
        cursor.execute(query, article_ids)
        rows = cursor.fetchall()

        articles_map = {}
        for row in rows:
            article_id = row["article_id"]
            if article_id not in articles_map:
                articles_map[article_id] = {
                    "id" : article_id,
                    "title" : row["title"],
                    "editor" : row["editor"],
                    "date" : row["date"],
                    "html" : row["html"],
                    "content" : []
                }
            raw_content = row["content"]
            articles_map[article_id]["content"].append(raw_content)
        
        formatted_result = ""
        for article_id, article_data in articles_map.items():
            full_body = "\n".join(article_data["content"])
            formatted_result += f"=== [ID:{article_id}] {article_data['title']} ===\n{full_body}\n\n"

        return formatted_result if formatted_result else "해당 ID들의 뉴스 본문을 찾을 수 없습니다다."

    except Exception as e:
        print(f"뉴스 본문 읽기 중 오류 발생 : {e}")
        return "뉴스 본문 읽기 실패"
    finally:
        conn.close()