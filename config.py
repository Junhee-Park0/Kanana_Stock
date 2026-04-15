import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path = _DOTENV_PATH)

class Config:
    """Agent 전역 설정"""
    # ============================================================================
    # 로깅 설정 (기본값)
    # ============================================================================
    ENABLE_LOCAL_LOGGING = True

    # ============================================================================
    # 모델 설정
    # ============================================================================
    KANANA_MODEL_NAME = "kakaocorp/kanana-1.5-2.1b-instruct-2505"
    KANANA_MAX_NEW_TOKENS = int(os.getenv("KANANA_MAX_NEW_TOKENS", "512"))
    KANANA_SUMMARY_MAX_NEW_TOKENS = int(os.getenv("KANANA_SUMMARY_MAX_NEW_TOKENS", "2048"))

    # ============================================================================
    # 경로 설정
    # ============================================================================
    LOG_DIR = "./logs"
    KANANA_MODEL_PATH = "./Kanana_Model"

    NEWS_FILE_PATH = "./data/News"
    NEWS_DB_PATH = "./database/News"
    SEC_FILE_PATH = "./data/SEC"
    SEC_DB_PATH = "./database/SEC"

    MAX_NEWS_COUNT = 20 # 수집할 뉴스 최대 개수
    MAX_SEC_DAYS = 14 # 수집할 SEC 일수

    DEBATE_HISTORY_PATH = "./debate"

    # ============================================================================
    # 티커 매핑
    # ============================================================================
    TICKER_MAP = {
        "NVDA": "NVIDIA",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "LLY": "Eli Lilly",
        "BAC": "Bank of America",
        "KO": "Coca-Cola",
        "META": "Meta",
        "AAPL": "Apple",
        "GOOG": "Google",
        "AMZN": "Amazon",
    }

    @classmethod
    def get_config_summary(cls):
        """현재 설정 요약"""
        return {
            "로컬 로깅": "활성화" if cls.ENABLE_LOCAL_LOGGING else "비활성화",
            "모델": cls.KANANA_MODEL_NAME,
        }