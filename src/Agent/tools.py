from typing import List, Dict
from pathlib import Path
from langchain_core.tools import tool

from src.Crawling.get_context import GetContext
from config import Config

NEWS_FILE_PATH = Config.NEWS_FILE_PATH
NEWS_DB_PATH = Config.NEWS_DB_PATH
MAX_NEWS_COUNT = Config.MAX_NEWS_COUNT
SEC_DB_PATH = Config.SEC_DB_PATH

MAX_NEWS_COUNT = Config.MAX_NEWS_COUNT
MAX_SEC_DAYS = Config.MAX_SEC_DAYS

context_manager = GetContext()

@tool
def search_recent_news(ticker : str, limit : int = MAX_NEWS_COUNT) -> str:
    """
    사용자 입력으로 받은 ticker에 따라 최신 뉴스 목록을 조회합니다. 
    뉴스의 전체 내용을 읽기 전, 어떤 사건들이 있었는지 확인할 때 사용합니다.
    제목과 날짜, 기사 ID 등을 확인하여 중요 뉴스를 선별합니다.
    """
    return context_manager.get_recent_news(ticker, limit)

@tool
def search_recent_filings(ticker : str, days : int = MAX_SEC_DAYS) -> str:
    """
    사용자 입력으로 받은 ticker에 따라 최신 SEC 공시 목록을 조회합니다.
    본문을 읽기 전, 어떤 공시가 있었는지 목록을 확인할 때 사용합니다.
    기업의 공식 재무 보고서나 중요 변동 사항이 있었는지 확인합니다.
    """
    return context_manager.get_recent_filings(ticker, days)

@tool
def read_news_content(article_id : str) -> str:
    """
    본문을 읽고자 하는 기사 ID(article_id)에 해당하는 뉴스 본문을 조회합니다.
    목록을 확인한 후, 원하는 뉴스의 상세한 맥락이나 표 데이터를 분석할 때 사용합니다.
    """
    return context_manager.read_news_content(article_id)

@tool
def read_parsed_filing(file_path : str) -> str:
    """
    본문을 읽고자 하는 file_path에 해당하는 SEC 공시 본문을 조회합니다.
    재무 수치, 리스크 요인 등 상세 공시 내용을 분석할 때 사용합니다. 
    """
    return context_manager.read_parsed_filing(file_path)