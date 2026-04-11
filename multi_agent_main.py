from typing import List
from src.Multi_Agent.graph import agent_debate_graph
from src.Multi_Agent.states import DebateAgentState
from src.Multi_Agent.functions import get_context
import os, argparse

def main(ticker : str, keywords : List[str]):
    print(f"{'='*60}")
    print(f"🔍 [Multi Agent] 토론 시작")
    print(f"{'='*60}")

    # 데이터 수집
    context = get_context(ticker, keywords)
    print(f"--- 데이터 수집 완료 ---")

    # 초기 상태 설정 
    initial_state = {
        "ticker" : ticker,
        "keywords" : keywords,
        "context" : context,
        "optimist_initial" : "",
        "pessimist_initial" : "",
        "debate_history" : [],
        "turn_count" : 0,
        "max_turns" : 6,
        "current_agent" : "start",
        "final_consensus" : None
    }

    # 그래프 생성 및 실행
    print("--- 멀티 에이전트 토론 시작 ---")
    graph = agent_debate_graph()
    result = graph.invoke(initial_state)

    # 결과 출력
    print("\n===========================================")
    print("               최종 합의안")
    print("===========================================")
    print(result["final_consensus"])
    print("===========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Multi Agent Debate")
    parser.add_argument("--ticker", type = str, required = True, help = "타겟 기업명 (Ticker)")
    parser.add_argument("--keywords", type = str, required = True, help = "키워드 (예. AI, Cloud, Gemini)")
    args = parser.parse_args()
    main(args.ticker, args.keywords)