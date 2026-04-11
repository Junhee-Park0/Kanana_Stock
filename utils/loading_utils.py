import json
from pathlib import Path
from typing import Dict

def load_ticker_map(config_path : str = "config") -> Dict[str, str]:
    """config/ticker_map.json 파일 로드하여 딕셔너리로 반환"""       
    file_path = Path(config_path) / "ticker_map.json"

    if not file_path.exists():
        print(f"Error: {file_path} not found")
    
    try:
        with open(file_path, "r", encoding = "utf-8") as f:
            return json.load(f)
    
    except Exception as e:
        print(f"Error: {e}")
        return {}