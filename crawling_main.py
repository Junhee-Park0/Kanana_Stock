from src.Crawling.news_crawling import News_Crawler
from src.Crawling.news_db import News_Database
from src.Crawling.sec_crawling import SEC_Crawler
from src.Crawling.sec_db import SEC_Database

import sys, os
import argparse 
from pathlib import Path

from config import Config
NEWS_FILE_PATH = Config.NEWS_FILE_PATH
NEWS_DB_PATH = Config.NEWS_DB_PATH
SEC_FILE_PATH = Config.SEC_FILE_PATH
SEC_DB_PATH = Config.SEC_DB_PATH

tickers = Config.TICKER_MAP

def ensure_directory(path : Path):
    """디렉토리가 없으면 생성하는 함수"""
    if not path.exists():
        path.mkdir(parents = True, exist_ok = True)

def main(ticker : str):
    """
    뉴스 + SEC 크롤링 및 DB 저장 함수
    Args: 
        ticker : 기업 티커 (예. NVDA)
    """
    ticker = ticker.upper()
    print(f"\n{'='*60}")
    print(f"🔍 [{ticker}] 뉴스 + SEC 데이터 수집기 시작")
    print(f"{'='*60}")

    # 1. DB 경로 설정
    news_db_dir = Path(NEWS_DB_PATH)
    sec_db_dir = Path(SEC_DB_PATH)
    ensure_directory(news_db_dir)
    ensure_directory(sec_db_dir)
    
    news_db_path = Path(NEWS_DB_PATH) / f"{ticker}"
    sec_db_path = Path(SEC_DB_PATH) / f"{ticker}"

    # 2. 뉴스 크롤러 및 DB
    news_crawler = News_Crawler()
    news_db = News_Database()

    print(f"\n[뉴스] 뉴스 데이터 수집 시작...")

    success, new_html_paths = news_db.crawl_and_update_news_db(ticker, news_db_path)
    if success:
        print(f"✅ 뉴스 데이터 수집 완료: {len(new_html_paths)}개 뉴스")
    else:
        print(f"❌ 뉴스 데이터 수집 실패..")

    # 3. SEC 크롤러 및 DB
    sec_crawler = SEC_Crawler()
    sec_db = SEC_Database()
    
    print(f"\n[SEC] SEC 데이터 수집 시작...")

    success, new_sec_paths = sec_db.crawl_and_update_sec_db(ticker, sec_db_path)
    if success:
        print(f"✅ SEC 데이터 수집 완료: {len(new_sec_paths)}개 공시")
    else:
        print(f"❌ SEC 데이터 수집 실패..")

    print(f"\n{'='*60}")
    print(f"🔍 [{ticker}] 뉴스 + SEC 데이터 수집기 완료")
    print(f"{'='*60}")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description = "뉴스 + SEC 데이터 수집기")
    parser.add_argument("--ticker", type = str, help = "기업 티커 (예. NVDA)")
    args = parser.parse_args()
    main(args.ticker)
