from src.Agent.graph import agent_debate_graph
import argparse
import traceback

def main(ticker : str):
    print(f"{'='*60}")
    print(f"🔍 [Multi Agent] {ticker} 분석 시작")
    print(f"{'='*60}")

    # 초기 상태 설정 
    initial_state = {
        "ticker" : ticker.upper(),
        "context" : "", # 에이전트가 tool로 업데이트할 공간
        "optimist_initial" : "",
        "pessimist_initial" : "",
        "debate_history" : [],
        "turn_count" : 0,
        "max_turns" : 6,
        "current_agent" : "start",
        "final_consensus" : None
    }

    # 그래프 생성 및 실행
    print("--- 🚀 Multi Agent Debate 시작 ---")
    graph = agent_debate_graph()
    try:
        result = graph.invoke(initial_state)

        # 결과 출력
        print("\n" + "="*50)
        print(f"🏆 {ticker} 투자 분석 최종 합의안")
        print("="*50)
        print(result.get("final_consensus", "합의안 도출에 실패했습니다.."))
        print("="*50)

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {repr(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Multi Agent Debate")
    parser.add_argument("--ticker", type = str, required = True, help = "타겟 기업명 (Ticker)")
    args = parser.parse_args()

    main(args.ticker.upper())