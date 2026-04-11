from typing import List, Dict, Literal
from pathlib import Path
import json, yaml
import os, sys
from src.Fetch_Data.get_context import GetContext
from src.Multi_Agent.states import DebateAgentState

def load_prompts(prompt_name : str) -> str:
    prompt_path = Path("src/Multi_Agent/prompts.yaml")
    with open(prompt_path, "r", encoding = 'utf-8') as f:
        prompts = yaml.safe_load(f)
    prompt = prompts[prompt_name]
    return prompt

def get_context(ticker : str, keywords: List[str]) -> str:
    """
    새로운 뉴스 기사들/SEC 데이터를 가져옵니다.
    Args: 
        ticker : 종목 코드
        keywords : 키워드 리스트
    Returns:
        context : 새로운 뉴스 기사들/SEC 데이터 (내용을 포함한 리스트)
    """
    gc = GetContext(ticker, keywords)
    news_context, sec_context = gc.get_context()
    context = news_context + sec_context    
    return context

def should_continue(state: DebateAgentState) -> Literal["optimist", "summary"]:
    """
    토론을 계속할지 중재자로 넘어갈지 결정하는 조건부 엣지 함수
    """
    if state["turn_count"] >= state["max_turns"]:
        return "summary"
    return "optimist"

def save_conversation_history(ticker : str, conversation_history : str) -> None:
    """
    Agent들의 대화 기록을 txt 파일로 저장합니다. 
    Args:
        conversation_history : Agent들의 대화 기록
    Returns:
        None
    """
    path = Path(f"data/debate/{ticker}_conversation_history.txt")
    if not path.exists():
        path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(f"Conversation History: \n\n{conversation_history}\n")  
    return None

def save_final_consensus(ticker : str, final_consensus : str) -> None:
    """
    Agent들의 최종 합의 결과를 txt 파일로 저장합니다. 
    Args:
        final_consensus : Agent들의 최종 합의 결과
    Returns:
        None
    """ 
    path = Path(f"data/debate/{ticker}_final_consensus.txt")
    if not path.exists():
        path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(f"Final Consensus: \n\n{final_consensus}\n")
    return None

