from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

import requests, os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

USER_EMAIL = os.getenv("USER_EMAIL")

from config import Config
SEC_FILE_PATH = Config.SEC_FILE_PATH
SEC_DB_PATH = Config.SEC_DB_PATH
MAX_SEC_DAYS = Config.MAX_SEC_DAYS

from src.Crawling.sec_parsing import SEC_Parser

class SEC_Crawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_EMAIL or "jhpark0256@korea.ac.kr"})

    def get_cik_from_ticker(self, ticker : str) -> Optional[str]:
        """티커로부터 CIK 코드를 반환"""
        try:
            url = f"https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url)
            response.raise_for_status()

            companies = response.json()

            ticker = ticker.upper()
            for company in companies.values():
                if company.get("ticker", "").upper() == ticker:
                    cik = str(company.get("cik_str")).zfill(10)
                    print(f"CIK 조회 성공 - {ticker}: {cik}")
                    return cik

            print(f"CIK 조회 실패 - 일치하는 티커 없음: {ticker}")
            return None
            
        except Exception as e:
            print(f"CIK 조회 중 오류 발생 - {ticker}: {e}")
            return None

    def get_sec_filings(self, cik : str, dates = MAX_SEC_DAYS) -> pd.DataFrame:
        """CIK 코드로부터 최근 14일간의 공시 보고서 반환"""
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.session.get(url)
            response.raise_for_status()
            sec_json = response.json()
            sec_data = pd.DataFrame(sec_json['filings']['recent'])
            sec_data['filingDate'] = pd.to_datetime(sec_data['filingDate'])

            # 10-K, 10-Q가 최근에 있었다면 포함
            k_filing = sec_data[sec_data['form'] == '10-K'].head(1)
            q_filings = sec_data[sec_data['form'] == '10-Q'].head(2)

            # 8-K, SC 13G, 4, DEF 14A는 최신 몇 개만 포함
            threshold_date = datetime.now() - timedelta(days = dates)
            recent_events = sec_data[
                (sec_data['filingDate'] >= threshold_date) &
                (sec_data['form'].isin(['8-K', 'SC 13G', '4', 'DEF 14A']))
            ]

            final_data = pd.concat([k_filing, q_filings, recent_events]).drop_duplicates(subset = ["accessionNumber"]).reset_index(drop = True)
            final_data = final_data.sort_values(by = 'filingDate', ascending = False)
            return final_data
            
        except Exception as e:
            print(f"CIK 조회 중 오류 발생 - {cik}: {e}")
            return None
    
    def _set_file_priorities(self, items : List[Dict], form : str) -> str:
        """파일 우선순위 결정 (파일명, 파일 크기 고려)"""
        selected_files = []
        
        for item in items:
            file_name = item.get("name", "").lower()

            try:
                file_size = int(item.get("size", 0))
            except ValueError:
                file_size = 0

            if form in ["10-Q", "10-K"]:
                if file_name.endswith("_htm.xml"):
                    selected_files.append((100, file_size, item["name"]))
                elif file_name.endswith((".htm", ".html")):
                    selected_files.append((50, file_size, item["name"]))

            elif form == "8-K":
                if "8k" in file_name and file_name.endswith((".htm", ".html")):
                    selected_files.append((100, file_size, item["name"]))

            elif form == "4":
                if "form4" in file_name and file_name.endswith(".xml"):
                    selected_files.append((100, file_size, item["name"]))

            elif form == "SC 13G":
                if "13g" in file_name and file_name.endswith((".xml", ".htm", ".html")):
                    selected_files.append((100, file_size, item["name"]))

            elif form == "DEF 14A":
                if "def14a" in file_name and file_name.endswith((".htm", ".html")):
                    selected_files.append((100, file_size, item["name"]))

        selected_files.sort(key = lambda x: (x[0], x[1]), reverse = True)

        return selected_files[0][2] if selected_files else None

    def download_filing_file(self, ticker : str, cik : str, accession_number : str, form : str) -> Optional[Path]:
        """공시 문서 파일 다운로드 후 저장 (Raw 폴더)"""
        true_cik = str(int(cik))
        accession_number = accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{true_cik}/{accession_number}/index.json"
        response = self.session.get(url)
        response.raise_for_status()
        index_json = response.json()

        file_name = self._set_file_priorities(index_json['directory']['item'], form)

        if file_name:
            try:
                url = f"https://www.sec.gov/Archives/edgar/data/{true_cik}/{accession_number}/{file_name}"
                response = self.session.get(url)
                response.raise_for_status()
                if response.status_code == 200:
                    file_path = f"{SEC_FILE_PATH}/{ticker}/Raw"
                    os.makedirs(file_path, exist_ok = True)
                    saved_file_path = Path(file_path) / file_name
                    with open(saved_file_path, "wb") as f:
                        f.write(response.content)
                    print(f"파일 다운로드 성공 - {file_name}")
                    return saved_file_path
            except Exception as e:
                print(f"파일 다운로드 중 오류 발생 - {file_name}: {e}")
                return None
        else:
            print(f"파일 다운로드 실패 - {file_name}")
            return None

    def parse_filing(self, ticker : str, file_path : Path, form : str) -> Optional[Path]:
        """다운받은 파일을 파싱하여 저장 (Parsed 폴더)"""
        sec_parser = SEC_Parser(ticker, file_path)
        return sec_parser.parse_filing(form)

    def download_and_parse_filing(self, ticker : str) -> Optional[Path]:
        """[최종] 티커를 입력받아 공시 문서 파일을 다운로드하고 파싱하여 저장"""
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            print("CIK 조회 실패로 SEC 수집을 종료합니다.")
            return None
        sec_filings_df = self.get_sec_filings(cik)
        if sec_filings_df is None or sec_filings_df.empty:
            print("수집할 SEC 공시가 없습니다.")
            return None
        print(sec_filings_df)

        for index, row in sec_filings_df.iterrows():
            file_path = self.download_filing_file(ticker, cik, row["accessionNumber"], row["form"])
            if file_path:
                parsed_file_path = self.parse_filing(ticker, file_path, row["form"])
                if parsed_file_path:
                    print(f"파싱 완료 - {index + 1}번째 파일 : {parsed_file_path}")
        return None

if __name__ == "__main__":
    crawler = SEC_Crawler()
    crawler.download_and_parse_filing("NVDA")
