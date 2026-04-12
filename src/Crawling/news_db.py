### 앞에서 crawling.py 의 결과로 
### scraped_news = scraped_news.append({'metadata' : metadata, 'content' : scraped_content}) 를 받아옴
### List[{Dict, List['str', pd.DataFrame]}]

import sqlite3
import pandas as pd
import os
from src.Crawling.news_crawling import News_Crawler
from typing import Dict, List

from selenium import webdriver
from pathlib import Path
from datetime import datetime
import re

import warnings
warnings.filterwarnings('ignore')

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")

class News_Database:
    def _init_db(self, conn: sqlite3.Connection):
        """SQLite 테이블/인덱스 생성"""
        cursor = conn.cursor()
        # 테이블 생성
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Articles(
                article_id INTEGER PRIMARY KEY AUTOINCREMENT, /* 각 기사의 고유 Index */
                ticker TEXT, /* 티커 */
                html TEXT UNIQUE NOT NULL, /* HTML 주소 */
                title TEXT, /* 제목 */
                editor TEXT, /* 에디터 */
                date TEXT, /* 날짜 */
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP /* 생성 날짜 */
                )
            """
        )
        # 날짜 컬럼에 인덱스 추가 (조회 성능 향상)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_articles_date ON Articles(date)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Content(
                block_id INTEGER PRIMARY KEY AUTOINCREMENT, /* 각 블록의 고유 Index */
                article_id INTEGER NOT NULL, /* 기사의 고유 Index를 참조 */
                block_order INTEGER NOT NULL, /* 블록의 순서 (1부터 시작) */
                block_type TEXT NOT NULL, /* 블록의 타입 (text, table) */
                content TEXT NOT NULL, /* 블록의 내용 */
                FOREIGN KEY (article_id) REFERENCES Articles (article_id) /* 두 테이블을 연결 */
                ON DELETE CASCADE /* 기사가 삭제되면 content 블록도 삭제 */
                )
            """
        )
        conn.commit()

    # 날짜순으로 정렬 (최신이 먼저 오도록) - 날짜 파싱을 위해 datetime 사용          
    def _parse_date(self, date_str):
        """
        날짜 문자열을 파싱하여 정렬 가능한 형식으로 변환
        지원 형식:
        - "Thu, November 13, 2025 at 3:45 AM GMT+9"
        - "Nov 27, 2024"
        - "November 27, 2024"조
        - "2024-11-27" 등
        """
        if not date_str:
            return datetime.min
        try:
            date_str = date_str.strip()
            
            # "X hours ago", "X days ago" 같은 상대적 날짜 처리
            if "ago" in date_str.lower():
                return datetime.now()
            
            # 요일과 시간 부분 제거
            if " at " in date_str:
                # "Thu, November 13, 2025 at 3:45 AM GMT+9" -> "November 13, 2025"
                date_part = date_str.split(" at ")[0]
                # 요일 제거 (예: "Thu, " 제거)
                date_part = re.sub(r'^[A-Za-z]+,?\s*', '', date_part)
                date_str = date_part.strip()
            
            # 일반적인 날짜 형식들 시도
            formats = [
                "%B %d, %Y",      # November 13, 2025
                "%b %d, %Y",      # Nov 27, 2024
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # 파싱 실패 시 원본 문자열 반환
            return date_str
        except Exception:
            return datetime.min

    def save_data_to_db(self, ticker, scraped_news, db_path):
        """
        스크래핑된 데이터를 받아 SQLite DB로 생성
        - metadata -> 'metadata' 테이블 생성
        - content -> 'content' 테이블 생성 
        """
        conn = None
        try:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents = True, exist_ok = True)
            conn = sqlite3.connect(db_path)
            self._init_db(conn)
            cursor = conn.cursor()           

            saved_count = 0
            
            # 날짜순으로 정렬 (최신이 먼저)
            sorted_news = sorted(
                scraped_news,
                key = lambda x: self._parse_date(x.get('metadata', {}).get('date', '')),
                reverse = True
            )
            
            # 옮기기 
            for article_data in sorted_news:
                metadata = article_data['metadata']
                content = article_data['content']

                try:
                    html = metadata.get('html')
                    if not html:
                        continue
                    # Articles 테이블에 메타데이터 삽입
                    cursor.execute(
                        """
                        INSERT INTO Articles (ticker, html, title, editor, date)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            ticker,
                            html,
                            metadata.get('title'),
                            metadata.get('editor'),
                            metadata.get('date')
                        )
                    )

                    # article_id 가져오기
                    article_id = cursor.lastrowid

                    # Content 테이블에 본문 삽입 
                    for index, item in enumerate(content):
                        block_order = index + 1 # 1부터 시작하도록 설정

                        # 줄글 부분이면 그대로 삽입
                        if isinstance(item, str):
                            block_type = "text"
                            content_data = item
                        
                        # 데이터프레임 부분이면 json으로 바꿔서 삽입
                        elif isinstance(item, pd.DataFrame):
                            block_type = "table"
                            content_data = item.to_json(orient = "split")
                        
                        else:
                            continue
                            
                        cursor.execute(
                            """
                            INSERT INTO Content (article_id, block_order, block_type, content)
                            VALUES (?, ?, ?, ?)
                            """,
                            (article_id, block_order, block_type, content_data)
                        )
                    
                    print(f"DB 저장 성공 : {metadata['html']} (Article ID : {article_id})")
                    saved_count += 1

                # 기존에 DB 에 존재하는 기사라면
                except sqlite3.IntegrityError:
                    print(f"DB 저장 건너뛰기 (이미 존재) : {metadata['html']}")
                    continue
                
                except Exception as e:
                    print(f"DB 저장 중 오류 발생 (기사 {metadata['html']}) : {e}")
                    conn.rollback() # 이 기사에 대한 변경사항만 롤백

            conn.commit()
            print(f"총 {saved_count}개 기사 저장 완료")
            return True

        except Exception as e:
            print(f"DB 연결/저장 실패 : {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                conn.close()
                print("News DB 연결 종료")

    def compare_news_db(self, db_path, html_path) -> bool:
        """
        기존에 존재하는 db_path와 새롭게 수집한 html_paths를 비교하여 기존 db에 없는 html만을 반환
        """
        query = "SELECT 1 FROM Articles WHERE html = ?"
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # 쿼리 실행
            cursor.execute(query, (html_path,))
            # 결과 
            result = cursor.fetchone() # html 주소가 존재하면 1, 없으면 None
            return result is None # None이면 True, 즉 새로운 기사다 
        
        except sqlite3.Error as e:
            print(f"DB 조회 중 오류 발생 : {e}")
            return False
        
        finally:
            if conn:
                conn.close()

    def get_articles_sorted_by_date(self, db_path, limit = None) -> List[Dict]:
        """
        날짜순으로 정렬된 기사 조회 (최신이 먼저)
        Args:
            db_path: DB 파일 경로
            limit: 조회할 최대 개수 (None이면 전체)
        Returns:
            기사 리스트 (날짜순 정렬, 최신이 먼저)
        """
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT article_id, html, title, editor, date
                FROM Articles
                ORDER BY date DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        
        except sqlite3.Error as e:
            print(f"DB 조회 중 오류 발생 : {e}")
            return []
        
        finally:
            if conn:
                conn.close()

    def crawl_and_update_news_db(self, ticker, db_path):
        """
        뉴스 데이터를 수집하고 DB에 저장
        Returns:
            tuple: (success: bool, new_html_paths: list) - 성공 여부와 새로운 HTML 경로 리스트
        """
        news_crawler = News_Crawler()
        # 일단 데이터 수집해오고
        try:
            _, html_paths = news_crawler.get_news_html_all(ticker, chrome_options)
            if not html_paths:
                print("뉴스 HTML 경로를 수집하지 못했습니다. (사이트 구조/XPath 변경 가능)")
                return False, []
        except Exception as e:
            print(f"뉴스 수집 중 오류 발생 : {e}")
            return False, []

        # 모아둔 html이랑 기존 db 비교해서 새로운 뉴스만 본문 수집하기
        if os.path.exists(db_path): # 이미 존재하는 db라면, 즉 크롤링 결과가 이미 존재한다면
            new_html_paths = [path for path in html_paths if self.compare_news_db(db_path, path) == True]
            if len(new_html_paths) == 0:
                print("새로운 뉴스가 없습니다. 크롤링을 종료합니다.")
                return True, []  # 새로운 뉴스가 없는 것도 성공으로 간주
            print(f"새로운 뉴스 {len(new_html_paths)}개를 찾았습니다.")
            try:
                scraped_news = news_crawler.get_news_content(new_html_paths, chrome_options)
                news_crawler.save_news_content(ticker, scraped_news)
                success = self.save_data_to_db(ticker, scraped_news, db_path)
                return success, new_html_paths
            except Exception as e:
                print(f"뉴스 본문 수집 중 오류 발생 : {e}")
                return False, []

        else: # 새롭게 수집해오는 기업인 경우
            print("--- DB를 새롭게 생성합니다 ---")
            try:
                scraped_news = news_crawler.get_news_content(html_paths, chrome_options)
                news_crawler.save_news_content(ticker, scraped_news)
                success = self.save_data_to_db(ticker, scraped_news, db_path)
                if success:
                    return True, html_paths
                else:
                    return False, []
            except Exception as e:
                print(f"뉴스 본문 수집 중 오류 발생 : {e}")
                return False, []