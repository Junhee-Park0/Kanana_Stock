"""
뉴스 크롤링 후 브리핑 작성하는 에이전트 (단일 에이전트)
"""
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.Single_Agent.tools.news_tools import search_news_headlines, read_news_content
from src.Single_Agent.tools.functions import load_prompts
import argparse

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.0)
tools = [search_news_headlines, read_news_content]

system_prompt_org = load_prompts("system_prompt")
system_prompt = system_prompt_org["role"] + "\n\n" + system_prompt_org["instructions"]
agent_executor = create_react_agent(llm, tools)

def run_agent(ticker):
    print(f"--- Single Agent 시작 ({ticker}) ---")

    inputs = {
        "messages" : [
            SystemMessage(content = system_prompt),            
            HumanMessage(content = f"{ticker}에 대한 최신 뉴스 브리핑을 작성해줘")]
    }

    for chunk in agent_executor.stream(inputs, stream_mode = "values"):
        message = chunk["messages"][-1]

        if message.type == "ai":
            if message.tool_calls:
                # tool call을 한 경우
                tool_name = message.tool_calls[0]["name"]
                tool_args = message.tool_calls[0]["args"]
                print(f"▶▶ Tool Call: {tool_name} with args: {tool_args}")
            else:
                print(f"\n✅ [Final Briefing]\n {message.content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "시장 브리핑 에이전트 실행")

    parser.add_argument("--ticker", type = str, required = True, help = "타겟 기업명 (Ticker)")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    # 유효성 검사
    tickers = {"NVDA" : 'NVIDIA', "MSFT" : 'Microsoft', "TSLA" : 'Tesla', 
           "LLY" : 'Eli Lilly', "BAC" : 'Bank of America', "KO" : 'Coca-Cola'
           }
    if ticker not in tickers:
        print(f"❌ 올바르지 않은 티커입니다. 티커를 다시 입력해주세요.")
        exit(1)

    run_agent(ticker)
