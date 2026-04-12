import os
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage

from datetime import datetime
from pathlib import Path
from config import Config

from dotenv import load_dotenv
load_dotenv(".env")

from src.Agent.kanana_pipeline import ChatKanana, get_kanana_pipeline
from src.Agent.states import DebateAgentState
from src.Agent.functions import load_prompt, create_agent
from src.Agent.tools import search_recent_news, search_recent_filings, read_news_content, read_parsed_filing


def optimistic_initial_node(state : DebateAgentState):
    """
    낙관론자 에이전트: 긍정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    print(f"\n[낙관론자] 초기 의견 도출 중...")
    llm = ChatKanana()
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("optimist_prompt")
    # 에이전트 실행기 생성
    agent_executor = create_agent(llm, tools, system_prompt)
    # 입력 메시지 구성
    input_message = f"""
    현재 {state['ticker']} 종목에 대한 낙관적 분석 의견을 제시해줘. 
    반드시 제공된 도구를 사용해서 최신 수치와 기사 내용을 근거로 들어야 해.
    """
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": state["ticker"],
        "input": input_message,
        "chat_history": []
    })
    print(f"[낙관론자] tool calls: {len(response.get('tool_calls', []))}")
    # 결과
    print(f"낙관론자: {response['output']}")
    return {
        "optimist_initial" : response["output"]
    }

def pessimistic_initial_node(state : DebateAgentState):
    """
    비관론자 에이전트: 부정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    print(f"\n[비관론자] 초기 의견 도출 중...")
    llm = ChatKanana()
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("pessimist_prompt")
    # 에이전트 실행기 생성
    agent_executor = create_agent(llm, tools, system_prompt)
    # 입력 메시지 구성
    input_message = f"""
    현재 {state['ticker']} 종목에 대한 비관적 분석 의견을 제시해줘. 
    반드시 제공된 도구를 사용해서 최신 수치와 기사 내용을 근거로 들어야 해.
    """
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": state["ticker"],
        "input": input_message,
        "chat_history": []
    })
    print(f"[비관론자] tool calls: {len(response.get('tool_calls', []))}")
    # 결과
    print(f"비관론자: {response['output']}")
    return {
        "pessimist_initial" : response["output"]
    }

def optimistic_debate_node(state : DebateAgentState):
    """
    낙관론자 토론 진행 중 : 상대의 논리를 반박하고 긍정적인 근거를 보강
    """
    turn = state.get("turn_count", 0) 
    print(f"\n[낙관론자 (Turn: {turn})] ------------------")  
    llm = ChatKanana()
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("optimist_debate_prompt")
    # 에이전트 실행기 생성
    agent_executor = create_agent(llm, tools, system_prompt)
    # 토론 맥락 구성
    opponent_initial = state.get("pessimist_initial", "아직 의견이 없습니다.")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    # 입력 메시지 구성 ("반박")
    input_message = (
        f"대상 종목: {state['ticker']}\n\n"
        f"[상대방의 초기 의견]\n{opponent_initial}\n\n"
        f"[지난 토론 기록]\n{history_str}\n\n"
        "위 비관적인 의견을 검토하고, 이를 반박할 수 있는 긍정적인 지표나 뉴스를 도구를 사용해 검색하십시오."
        "그 후, 수치적 근거를 바탕으로 반박 논리를 7문장 내외로 작성해주세요."
    )
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": state["ticker"],
        "input": input_message,
        "chat_history": []
    })
    print(f"[낙관론자 Turn {turn}] tool calls: {len(response.get('tool_calls', []))}")
    # 결과
    content = response["output"]
    print(f"낙관론자: {content}")
    
    return {
        "debate_history" : [f"낙관론자(Turn {turn}): {content}"],
        "turn_count": turn + 1,
        "current_agent": "optimist"
    }

def pessimistic_debate_node(state : DebateAgentState):
    """
    비관론자 토론 진행 중 
    """
    turn = state.get("turn_count", 0) 
    print(f"\n[비관론자 (Turn: {turn})] ------------------")
    llm = ChatKanana()
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("pessimist_debate_prompt")
    # 에이전트 실행기 생성
    agent_executor = create_agent(llm, tools, system_prompt)
    # 토론 맥락 구성
    opponent_initial = state.get("optimist_initial", "아직 의견이 없습니다.")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    # 입력 메시지 구성 ("반박")
    input_message = (
        f"대상 종목: {state['ticker']}\n\n"
        f"[상대방의 초기 의견]\n{opponent_initial}\n\n"
        f"[지난 토론 기록]\n{history_str}\n\n"
        "위 낙관적인 의견을 검토하고, 이를 반박할 수 있는 부정적인 지표나 뉴스를 도구를 사용해 검색하십시오."
        "그 후, 수치적 근거를 바탕으로 반박 논리를 7문장 내외로 작성해주세요."
    )
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": state["ticker"],
        "input": input_message,
        "chat_history": []
    })
    print(f"[비관론자 Turn {turn}] tool calls: {len(response.get('tool_calls', []))}")
    # 결과
    content = response["output"]
    print(f"비관론자: {content}")
    return {
        "debate_history" : [f"비관론자(Turn {turn}): {content}"],
        "turn_count" : turn + 1,
        "current_agent" : "pessimist"
    }

def summary_node(state: DebateAgentState):
    """
    중재자 에이전트: 토론 내용을 종합하여 최종 결론을 내립니다.
    """
    print("\n[중재자] ------------------")
    
    # 프롬프트 로드
    system_prompt = load_prompt("neutral_prompt")
    # 토론 맥락 취합
    history_str = "\n\n".join(state.get("debate_history", []))
    # 입력 메시지 구성
    input_message = (
        f"대상 종목: {state['ticker']}\n\n"
        f"[낙관론자 초기 의견]\n{state.get('optimist_initial', '내용 없음.')}\n\n"
        f"[비관론자 초기 의견]\n{state.get('pessimist_initial', '내용 없음.')}\n\n"
        f"[지난 토론 기록]\n{history_str}\n\n"
        "위 토론 내용을 종합적으로 분석하여, 최종 결론을 내리십시오."
        "양쪽이 제시한 수치적 근거와 뉴스 데이터를 객관적으로 비교 분석하여, 어느 쪽의 논리가 더 설득력 있는지 판단하세요."
        "그 후, 투자자가 주의해야 할 핵심 포인트와 최종 투자 의견을 종합적인 관점에서 작성해주세요."
    )
    # 에이전트 실행 (tool call 필요 없으므로 일반 invoke 사용)
    llm = ChatKanana()
    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = input_message)
    ]
    # 결과
    response = llm.invoke(messages)
    content = response.content
    print(f"⚖️중재자⚖️: {content}")
    return {
        "final_consensus": content
    }

def save_debate_node(state : DebateAgentState):
    """
    토론 기록과 최종 결론을 txt 파일로 저장
    """
    print("\n[System] 결과 저장 중...")
    ticker = state["ticker"]
    # 전체 기록 구성
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    debate_path = Path(f"{Config.DEBATE_HISTORY_PATH}/{ticker}/{date_str}")
    if not debate_path.exists():
        debate_path.mkdir(parents = True, exist_ok = True)
    # 전체 기록 구성
    full_report = [
        f"=== {ticker} Multi-Agent Debate Report ===",
        f"Date: {date_str}",
        "\n[1. Optimist Initial Opinion]", state.get("optimist_initial", ""),
        "\n[2. Pessimist Initial Opinion]", state.get("pessimist_initial", ""),
        "\n[3. Debate History]", "\n".join(state.get("debate_history", [])),
        "\n[4. Final Consensus]", state.get("final_consensus", "No consensus reached."),
    ] 
    # 전체 기록 저장
    report_text = "\n".join(full_report)
    try:
        with open(debate_path / "full_report.txt", "w", encoding = "utf-8") as f:
            f.write(report_text)
        print(f"[System] '{ticker}' 토론 결과 저장 완료.")
    except Exception as e:
        print(f"[System] 토론 결과 저장 실패: {e}")
    return state   