import os
from dotenv import load_dotenv
load_dotenv(".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.Multi_Agent.states import DebateAgentState
from src.Multi_Agent.functions import load_prompts, save_conversation_history, save_final_consensus

llm = ChatOpenAI(model="gpt-4o", temperature = 0.5)

def optimistic_initial_node(state : DebateAgentState):
    """
    낙관론자 에이전트: 긍정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    print(f"\n[낙관론자] 초기 의견 도출 중...")
    prompt = load_prompts("optimist_prompt")
    system_content = f"{prompt['role']}\n{prompt['instructions']}"
    user_content = prompt['inputs'].format(ticker = state['ticker'], context = state['context'])
    messages = [
        SystemMessage(content = system_content),
        HumanMessage(content = user_content)
    ]
    response = llm.invoke(messages)
    content = response.content
    print(f"낙관론자: {content}")
    return {
        "optimist_initial" : content
    }

def pessimistic_initial_node(state : DebateAgentState):
    """
    비관론자 에이전트: 부정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    print(f"\n[비관론자] 초기 의견 도출 중...")
    prompt = load_prompts("pessimist_prompt")
    system_content = f"{prompt['role']}\n{prompt['instructions']}"
    user_content = prompt['inputs'].format(ticker = state['ticker'], context = state['context'])
    messages = [
        SystemMessage(content = system_content),
        HumanMessage(content = user_content)
    ]
    response = llm.invoke(messages)
    content = response.content
    print(f"비관론자: {content}")
    return {
        "pessimist_initial" : content
    }

def optimistic_debate_node(state : DebateAgentState):
    """
    낙관론자 토론 진행 중 
    """
    print(f"\n[낙관론자 (Turn: {state.get('turn_count', 0)})] ------------------")
    prompt = load_prompts("optimist_debate_prompt")
    system_content = f"{prompt['role']}\n{prompt['instructions']}"
    opponent_initial = state.get("pessimist_initial", "")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    user_content = prompt['inputs'].format(ticker = state['ticker'], context = state['context'], history = history_str, opponent_initial = opponent_initial)
    messages = [
        SystemMessage(content = system_content),
        HumanMessage(content = user_content)
    ]
    response = llm.invoke(messages)
    content = response.content

    print(f"낙관론자: {content}")
    
    return {
        "debate_history" : [f"낙관론자: {content}"],
        "turn_count": state["turn_count"] + 1,
        "current_agent": "optimist"
    }

def pessimistic_debate_node(state : DebateAgentState):
    """
    비관론자 토론 진행 중 
    """
    print(f"\n[비관론자 (Turn: {state.get('turn_count', 0)})] ------------------")
    prompt = load_prompts("pessimist_debate_prompt")
    system_content = f"{prompt['role']}\n{prompt['instructions']}"
    opponent_initial = state.get("optimist_initial", "")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    user_content = prompt['inputs'].format(ticker = state['ticker'], context = state['context'], history = history_str, opponent_initial = opponent_initial)
    messages = [
        SystemMessage(content = system_content),
        HumanMessage(content = user_content)
    ]
    response = llm.invoke(messages)
    content = response.content

    print(f"비관론자: {content}")

    return {
        "debate_history" : [f"비관론자: {content}"],
        "turn_count" : state["turn_count"] + 1,
        "current_agent" : "pessimist"
    }

def summary_node(state: DebateAgentState):
    """
    중재자 에이전트: 토론 내용을 종합하여 최종 결론을 내립니다.
    """
    print("\n[중재자] ------------------")
    
    # 프롬프트 로드
    prompt = load_prompts("neutral_prompt")
    system_content = f"{prompt['role']}\n{prompt['instructions']}"
    
    # 입력 데이터 포맷팅
    context_str = state['context']
    history_str = "\n".join(state['debate_history']) if state['debate_history'] else "없음"
    
    # 프롬프트 구성
    user_content = prompt['inputs'].format(ticker = state['ticker'], context = context_str, history = history_str, optimist_initial = state['optimist_initial'], pessimist_initial = state['pessimist_initial'])
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]
    
    # LLM 호출
    response = llm.invoke(messages)
    content = response.content
    
    print(f"\n=== 최종 결론 ===\n{content}\n")
    
    return {
        "final_consensus": content
    }

def save_debate_node(state : DebateAgentState):
    """
    토론 기록과 최종 결론을 txt 파일로 저장
    """
    print("\n[System] 결과 저장 중...")
    ticker = state["ticker"]
    history = "\n".join(state["debate_history"])
    consensus = state.get("final_consensus", "No consensus reached.")

    save_conversation_history(ticker, history)
    save_final_consensus(ticker, consensus)
    print(f"[System] '{ticker}' 토론 결과 저장 완료.")
    return state