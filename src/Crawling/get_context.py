"""
DB에서 뉴스/공시 데이터를 조회하는 클래스 (tool의 기반이 되는 함수)
"""

import sqlite3
from pathlib import Path
import json
from typing import List, Dict
from datetime import datetime, timedelta

from config import Config

NEWS_FILE_PATH = Config.NEWS_FILE_PATH
NEWS_DB_PATH = Config.NEWS_DB_PATH
SEC_FILE_PATH = Config.SEC_FILE_PATH
SEC_DB_PATH = Config.SEC_DB_PATH

MAX_NEWS_COUNT = Config.MAX_NEWS_COUNT
MAX_SEC_DAYS = Config.MAX_SEC_DAYS

class GetContext:
    def __init__(self, news_db_path : str = NEWS_DB_PATH, sec_db_path : str = SEC_DB_PATH):
        self.news_db_path = Path(news_db_path)
        self.sec_db_path = Path(sec_db_path)

    def _resolve_ticker_db_path(self, base_path: Path, ticker: str) -> Path:
        """
        티커별 DB 파일 경로를 반환.
        기존 저장 규칙(/database/.../<TICKER>)을 우선 사용하고,
        없으면 .db 확장자 파일도 허용합니다.
        """
        ticker = ticker.upper()
        direct_path = base_path / ticker
        if direct_path.exists():
            return direct_path

        db_path = base_path / f"{ticker}.db"
        return db_path

    def _run_query(self, db_path : str, query : str, params : tuple) -> List[Dict]:
        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"DB 조회 중 오류 발생: {e}")
            return []

    def get_recent_news(self, ticker : str, limit : int = MAX_NEWS_COUNT) -> List[Dict]:
        """최신 뉴스 목록 조회 (항상 최신 10개)"""
        del limit
        limit = 10
        db_path = self._resolve_ticker_db_path(self.news_db_path, ticker)
        query = """
            SELECT article_id, html, title, editor, date
            FROM Articles
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        return self._run_query(db_path, query, (ticker.upper(), limit))

    def get_recent_filings(self, ticker : str, days : int = MAX_SEC_DAYS) -> List[Dict]:
        """SEC 공시 목록 조회 (최근 일수 공시 OR 10-K/10-Q/8-K)"""
        db_path = self._resolve_ticker_db_path(self.sec_db_path, ticker)
        since_date = (datetime.now() - timedelta(days = days)).strftime("%Y-%m-%d")
        query = """
            SELECT filing_id, parsed_path, file_name, form, filed_date
            FROM Filings
            WHERE ticker = ?
              AND (
                    filed_date >= ?
                    OR form IN ('10-K', '10-Q', '8-K')
                  )
            ORDER BY filed_date DESC
        """
        return self._run_query(db_path, query, (ticker.upper(), since_date))


    def read_news_content(self, article_id : str) -> str:
        """뉴스 Content 테이블의 문장 조각들을 하나의 기사 본문으로 병합"""
        query = """
            SELECT content, block_type
            FROM Content
            WHERE article_id = ?
            ORDER BY block_order ASC
        """
        blocks: List[Dict] = []

        # 티커 정보 없이 article_id만 받으므로, 뉴스 DB 디렉토리 내 파일을 순회하며 조회
        if self.news_db_path.exists():
            for db_file in self.news_db_path.iterdir():
                if not db_file.is_file():
                    continue
                blocks = self._run_query(db_file, query, (article_id,))
                if blocks:
                    break

        if not blocks:
            return "본문 내용을 찾을 수 없습니다."
        
        full_text = "\n\n".join([block["content"] for block in blocks])
        return full_text 

    def read_parsed_filing(self, file_path : str) -> Dict:
        """에이전트가 특정 SEC 공시 파일을 읽고 싶을 때 호출"""
        path = Path(file_path)
        if path.exists():
            with open(path, "r", encoding = "utf-8") as f:
                return json.load(f)
        return {"error": "File not found"}

