import argparse

from src.Crawling.crawling_main import main as run_crawling
from src.Agent.agent_main import main as run_agent_debate


def main(ticker: str):
    ticker = ticker.upper()

    print("\n" + "=" * 60)
    print(f"🚀 [{ticker}] 통합 파이프라인 시작 (Crawling + Debate)")
    print("=" * 60)

    print("\n[1/2] 데이터 크롤링 및 DB 업데이트 시작")
    run_crawling(ticker)

    print("\n[2/2] Multi Agent 토론 시작")
    run_agent_debate(ticker)

    print("\n" + "=" * 60)
    print(f"✅ [{ticker}] 통합 파이프라인 완료")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="크롤링과 멀티 에이전트 토론을 순차 실행하는 통합 메인"
    )
    parser.add_argument("--ticker", type=str, required=True, help="기업 티커 (예: NVDA)")
    args = parser.parse_args()

    main(args.ticker)
