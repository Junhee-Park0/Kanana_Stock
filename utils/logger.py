import logging
import sys
from datetime import datetime
from pathlib import Path
from config import Config

class RealTimeFileHandler(logging.FileHandler):
    """실시간으로 로그가 기록되도록 하는 핸들러"""
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(name: str = "Stock_Agent"):
    """로거 설정"""
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 파일 핸들러
    if Config.ENABLE_LOCAL_LOGGING:
        log_dir = Path(Config.LOG_DIR)
        log_dir.mkdir(exist_ok = True)
        
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = log_dir / f"stock_agent_{now_str}.log"
        file_handler = RealTimeFileHandler(log_filename, encoding = 'utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 기본 로거 인스턴스
logger = setup_logger()

def log_conversation(user_message: str, ai_response: str, session_id: str = None):
    """대화 로그 기록"""
    logger.info(f"[CONVERSATION] Session: {session_id}")
    logger.info(f"[USER] {user_message}")
    logger.info(f"[LLM] {ai_response}")

def log_error(error: Exception, context: str = ""):
    """에러 로그 기록"""
    logger.error(f"[ERROR] {context}: {str(error)}", exc_info = True) # 에러 난 위치 저장

def log_agent_action(action: str, details: dict = None):
    """Agent 액션 로그 기록"""
    log_message = f"[AGENT] {action}"
    if details:
        log_message += f" - Details: {details}"
    logger.info(log_message)

def log_tool_call(step: int, tool_name: str, args: dict, result_count = None):
    """Tool Call 로그 기록"""
    import json
    step_label = "auto" if step == 0 else str(step)
    try:
        args_str = json.dumps(args, ensure_ascii = False)
    except Exception:
        args_str = str(args)
    message = f"[TOOL_CALL - step {step_label}] tool = {tool_name} args = {args_str}"
    if result_count is not None:
        message += f" result_count = {result_count}"
    logger.info(message)
