import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.Crawling.sec_crawling import SEC_Crawler


class SEC_Database:
    def _init_db(self, conn: sqlite3.Connection):
        """SQLite 테이블/인덱스 생성"""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Filings(
                filing_id INTEGER PRIMARY KEY AUTOINCREMENT, /* 각 filing의 고유 Index */
                parsed_path TEXT UNIQUE NOT NULL, /* 파일 경로 */
                file_name TEXT NOT NULL, /* 파일 이름 */
                ticker TEXT, /* 티커 */
                form TEXT, /* 공시 형식 */
                reporter_name TEXT, /* 신고자 이름 */
                filed_date TEXT, /* 공시 날짜 */
                raw_metadata TEXT, /* 원본 json */
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP /* 생성 날짜 */
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_filings_date ON Filings(filed_date)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Content(
                block_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id INTEGER NOT NULL,
                block_order INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (filing_id) REFERENCES Filings(filing_id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_content_filing ON Content(filing_id)
            """
        )
        conn.commit()

    def _extract_filing(self, parsed_path: Path) -> Tuple[Dict, List[Tuple[str, str]]]:
        """파싱된 json 파일에서 필요한 부분 추출"""
        with open(parsed_path, "r", encoding = "utf-8") as f:
            data = json.load(f)

        document_type = data.get("document_type", "UNKNOWN")
        metadata = {
            "document_type": document_type,
            "form": document_type,
            "reporter_name": data.get("reporter_name"),
            "ticker": data.get("ticker"),
            "filed_date": data.get("period_of_report") or data.get("event_date") or data.get("period_end_date"),
            "raw_metadata": json.dumps(data, ensure_ascii = False)
        }

        blocks: List[Tuple[str, str]] = []
        if isinstance(data.get("transactions"), list):
            for tx in data["transactions"]:
                blocks.append(("transaction", json.dumps(tx, ensure_ascii = False)))

        if isinstance(data.get("reporting_persons"), list):
            for person in data["reporting_persons"]:
                blocks.append(("reporting_person", json.dumps(person, ensure_ascii = False)))

        if isinstance(data.get("text_chunks"), list):
            for text in data["text_chunks"]:
                if isinstance(text, str) and text.strip():
                    blocks.append(("text", text))

        if not blocks:
            blocks.append(("raw_json", json.dumps(data, ensure_ascii = False)))

        return metadata, blocks

    def save_data_to_db(self, parsed_paths: List[Path], db_path: Path) -> bool:
        """추출한 부분을 Filing/Content 테이블에 저장"""
        conn = None
        try:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents = True, exist_ok = True)
            conn = sqlite3.connect(db_path)
            self._init_db(conn)
            cursor = conn.cursor()

            saved_count = 0
            for parsed_path in parsed_paths:
                parsed_path = Path(parsed_path)
                if not parsed_path.exists():
                    print(f"SEC 저장 건너뜀 (파일 없음): {parsed_path}")
                    continue

                try:
                    metadata, blocks = self._extract_filing(parsed_path)
                    ticker_fallback = parsed_path.parent.parent.name.upper() if parsed_path.parent.parent else None
                    ticker_value = metadata["ticker"] or ticker_fallback
                    cursor.execute(
                        """
                        INSERT INTO Filings (
                            parsed_path, file_name, ticker, form, 
                            reporter_name, filed_date, raw_metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(parsed_path),
                            parsed_path.name,
                            ticker_value,
                            metadata["form"],
                            metadata["reporter_name"],
                            metadata["filed_date"],
                            metadata["raw_metadata"],
                        )
                    )
                    filing_id = cursor.lastrowid

                    for idx, (block_type, content) in enumerate(blocks, 1):
                        cursor.execute(
                            """
                            INSERT INTO Content (filing_id, block_order, block_type, content)
                            VALUES (?, ?, ?, ?)
                            """,
                            (filing_id, idx, block_type, content)
                        )
                    saved_count += 1
                    print(f"SEC DB 저장 성공: {parsed_path.name} (Filing ID: {filing_id})")

                except sqlite3.IntegrityError:
                    print(f"SEC DB 저장 건너뜀 (이미 존재): {parsed_path.name}")
                    continue
                except Exception as e:
                    print(f"SEC DB 저장 실패 ({parsed_path.name}): {e}")
                    conn.rollback()

            conn.commit()
            print(f"총 {saved_count}개 SEC 공시 저장 완료")
            return True

        except Exception as e:
            print(f"SEC DB 연결/저장 실패: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                conn.close()
                print("SEC DB 연결 종료")

    def compare_sec_db(self, db_path: Path, parsed_path: Path) -> bool:
        """이미 있는 기업이라면 새로운 공시가 있는지 확인"""
        query = "SELECT 1 FROM Filings WHERE parsed_path = ?"
        conn = None
        try:
            if not Path(db_path).exists():
                return True
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query, (str(parsed_path),))
            return cursor.fetchone() is None
        except sqlite3.Error as e:
            print(f"SEC DB 조회 오류: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_filings_sorted_by_date(self, db_path: Path, limit: Optional[int] = None) -> List[Dict]:
        """공시 날짜 기준 정렬"""
        conn = None
        try:
            if not Path(db_path).exists():
                return []
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT filing_id, parsed_path, file_name, ticker, form,
                       reporter_name, filed_date, created_at
                FROM Filings
                ORDER BY filed_date DESC, created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"SEC DB 조회 오류: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def crawl_and_update_sec_db(self, ticker: str, db_path: Path, dates: int = 14):
        """[최종] SEC 공시 데이터 수집, 파싱 후 중복 제거하여 DB에 저장"""
        crawler = SEC_Crawler()
        ticker = ticker.upper()
        # 데이터 수집
        cik = crawler.get_cik_from_ticker(ticker)
        if not cik:
            print("CIK 조회 실패로 SEC 수집을 종료합니다.")
            return False, []

        filings_df = crawler.get_sec_filings(cik, dates = dates)
        if filings_df is None or filings_df.empty:
            print("수집할 SEC 공시가 없습니다.")
            return True, []
        # 수집한 공시 파일 목록 파싱 후 저장
        parsed_paths: List[Path] = []
        for _, row in filings_df.iterrows():
            raw_file_path = crawler.download_filing_file(ticker, cik, row["accessionNumber"], row["form"])
            if not raw_file_path:
                continue
            try:
                parsed_path = crawler.parse_filing(ticker, raw_file_path, row["form"])
                if parsed_path:
                    parsed_paths.append(Path(parsed_path))
            except Exception as e:
                print(f"SEC 파싱 실패 ({raw_file_path}): {e}")

        if not parsed_paths:
            print("파싱된 SEC 파일이 없습니다.")
            return False, []
        # 이미 있는 기업이라면
        if os.path.exists(db_path):
            new_sec_paths = [p for p in parsed_paths if self.compare_sec_db(db_path, p)]
            if len(new_sec_paths) == 0:
                print("새로운 SEC 공시가 없습니다.")
                return True, []
            print(f"새로운 SEC 공시 {len(new_sec_paths)}개를 찾았습니다.")
            try:
                success = self.save_data_to_db(new_sec_paths, db_path)
                if success:
                    return True, new_sec_paths
                else:
                    return False, []
            except Exception as e:
                print(f"SEC 공시 저장 중 오류 발생 : {e}")
                return False, []
        else:
            print("--- DB를 새롭게 생성합니다 ---")
            try:
                success = self.save_data_to_db(parsed_paths, db_path)
                if success:
                    return True, parsed_paths
                else:
                    return False, []
            except Exception as e:
                print(f"SEC 공시 저장 중 오류 발생 : {e}")
                return False, []