import os

from datetime import datetime
from pathlib import Path
from config import Config

from src.Agent.kanana_pipeline import call_kanana_structured, extract_pure_text
from src.Agent.states import DebateAgentState
from src.Agent.functions import load_prompt, create_agent
from src.Agent.schemas import ConsensusOutput
from src.Agent.tools import search_recent_news, search_recent_filings, read_news_content, read_parsed_filing
from utils.logger import log_agent_action

def optimistic_initial_node(state : DebateAgentState):
    """
    낙관론자 에이전트: 긍정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    ticker = state["ticker"]
    log_agent_action("Optimist Initial Node Start", {"ticker": state["ticker"]})
    print(f"\n🙂 [낙관론자] 초기 의견 도출 중...")
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("optimist_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "initial")
    # 입력 메시지 구성
    input_message = f"""
    현재 {ticker} 종목에 대한 낙관적 분석 의견을 제시해줘. 
    반드시 제공된 도구를 사용해서 최신 수치와 기사 내용을 살펴보고, 이를 근거로 분석해야 해.
    """
    # 에이전트 실행 -> 여기가 payload
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": []
    })
    # 결과 출력
    clean_output = extract_pure_text(response.text)
    if len(clean_output) < 30:
        clean_output = response.text.strip()

    print(f"\n[낙관론자 답변]:\n{clean_output}")

    return {
        "optimist_initial" : clean_output,
        "tool_calls" : response.tool_calls
    }

def pessimistic_initial_node(state : DebateAgentState):
    """
    비관론자 에이전트: 부정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    ticker = state["ticker"]
    print(f"\n☹️ [비관론자] 초기 의견 도출 중...")
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("pessimist_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "initial")
    # 입력 메시지 구성
    input_message = f"""
    현재 {ticker} 종목에 대한 비관적 분석 의견을 제시해줘. 
    반드시 제공된 도구를 사용해서 최신 수치와 기사 내용을 살펴보고, 이를 근거로 분석해야 해.
    """
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": []
    })
    # 결과 출력
    clean_output = extract_pure_text(response.text)
    if len(clean_output) < 30:
        clean_output = response.text.strip()

    print(f"\n[비관론자 답변]:\n{clean_output}")

    return {
        "pessimist_initial" : clean_output,
        "tool_calls" : response.tool_calls
    }

def optimistic_debate_node(state : DebateAgentState):
    """
    낙관론자 토론 진행 중 : 상대의 논리를 반박하고 긍정적인 근거를 보강
    """
    turn = state.get("turn_count", 0) 
    ticker = state["ticker"]
    print(f"\n🙂 [낙관론자 (Turn: {turn})] ------------------")  
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("optimist_debate_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "debate")
    # 토론 맥락 구성
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    # 비관론자의 직전 의견 찾기
    last_opponent_message = "아직 상대방의 의견이 없습니다."
    if history_list:
        # 리스트를 뒤에서부터 훑으며 '비관론자'의 마지막 발언을 찾음
        for message in reversed(history_list):
            if "비관론자" in message:
                last_opponent_message = message
                break
    else:
        last_opponent_message = state.get("pessimist_initial", "비관론자의 초기 의견을 확인해주세요.")

    # 입력 메시지 구성 ("반박")
    input_message = (
        f"### 1. 당신의 정체성 ###\n"
        f"당신은 {ticker}의 강력한 지지자이자 매수론자입니다. 어떤 상황에서도 상승 논리를 펼쳐야 합니다.\n\n"
        f"### 2. 반드시 격파해야 할 상대방의 '비관적' 주장 ###\n"
        f"{last_opponent_message}\n\n"
        f"### 3. 전체 토론 기록 (참고용 - 중복 답변 방지) ###\n"
        f"{history_str}\n\n"
        f"### 4. 낙관론자 특별 지침 ###\n"
        f"- 위 비관론자의 주장은 틀렸음을 증명하십시오.\n"
        f"- {ticker}의 AI 파트너십과 매출 성장 가능성 등 '호재' 위주로만 답변하십시오.\n"
        f"- 절대로 비관적인 톤(위험하다, 거품이다 등)을 흉내 내지 마십시오.\n"
        f"- 위 [전체 토론 기록]에서 당신이 이미 했던 말이나 문장을 다시 쓰는 것은 금지됩니다.\n"
        f"- 반드시 새로운 뉴스 ID나 지표를 찾아내어 상대의 논리를 반박하십시오.\n"
        f"- 답변의 첫 문장은 반드시 상대방의 낙관적인 키워드를 인용하며 비판적으로 시작하십시오."
    )
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": [],
        "opponent_text": state.get("pessimist_initial", "")
    })
    print(f"[낙관론자 Turn {turn}] 분석 완료 (도구 사용: {len(response.tool_calls)}회)")
    # print(f"[상대 의견 요약]: {response.opponent_text[:50]}...")
    # 결과
    clean_output = extract_pure_text(response.text)
    if "상대 논리 인용" in clean_output or len(clean_output) < 20:
        clean_output = f"{ticker}의 핵심 지표와 최신 파트너십 뉴스를 종합할 때, 현재의 주가 하락 우려는 과도하며 장기적 성장 모멘텀은 여전히 견고합니다."

    new_history = f"낙관론자(Turn {turn}): {clean_output}"
    print(new_history)

    return {
        "debate_history" : [new_history],
        "turn_count": turn + 1,
        "current_agent": "optimist",
        "tool_calls" : response.tool_calls
    }

def pessimistic_debate_node(state : DebateAgentState):
    """
    비관론자 토론 진행 중 
    """
    turn = state.get("turn_count", 0) 
    ticker = state["ticker"]
    print(f"\n☹️ [비관론자 (Turn: {turn})] ------------------")
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("pessimist_debate_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "debate")
    # 토론 맥락 구성
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    last_opponent_message = "아직 상대방의 의견이 없습니다."
    if history_list:
        # 리스트를 뒤에서부터 훑으며 '낙관론자'의 마지막 발언을 찾음
        for message in reversed(history_list):
            if "낙관론자" in message:
                last_opponent_message = message
                break
    else:
        last_opponent_message = state.get("optimist_initial", "낙관론자의 초기 의견을 확인해주세요.")

    # 입력 메시지 구성 ("반박")
    input_message = (
        f"### 1. 당신의 정체성 ###\n"
        f"당신은 {ticker}의 주가 하락을 확신하는 사람입니다. 절대로 상대방의 의견에 동조하거나 칭찬하지 마십시오. "
        f"상대방의 근거 없는 희망 회로를 데이터로 파괴하십시오.\n"
        f"상대의 의견에 대해 '일리 있다', '동의한다' 같은 표현은 즉시 패배로 간주합니다.\n"
        f"### 2. 지금 당장 박살내야 할 상대방의 헛소리 ###\n"
        f"{last_opponent_message}\n\n"
        f"### 3. 전체 토론 기록 (참고용 - 중복 답변 방지) ###\n"
        f"{history_str}\n\n"
        f"### 4. 비관론자 특별 지침 ###\n"
        f"1. 시작: 반드시 상대방의 핵심 호재 키워드를 인용하며 '이는 전형적인 눈속임'임을 지적하며 시작하십시오.\n"
        f"2. 반박: 상대가 말한 근거들이 실제로는 '수익성 없는 비용 지출'임을 데이터로 증명하십시오.\n"
        f"3. 차별화: {history_str}에 이미 나온 근거는 쓰지 말고, 새로운 뉴스 ID를 사용하십시오.\n"
        f"4. 방식: 위 [전체 토론 기록]에서 당신이 이미 했던 말이나 문장을 쓰지 말고, 날카롭고 간결하게 독설을 퍼부으십시오."
    )
    # 에이전트 실행 
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": [],
        "opponent_text": state.get("optimist_initial", "")
    })
    print(f"[비관론자 Turn {turn}] 분석 완료 (도구 사용: {len(response.tool_calls)}회)")
    # print(f"[상대 의견 요약]: {response.opponent_text[:50]}...")
    # 결과
    clean_output = extract_pure_text(response.text)

    if "상대 논리 인용" in clean_output or len(clean_output) < 20:
        clean_output = f"{ticker}의 핵심 지표와 최신 파트너십 뉴스를 종합할 때, 현재의 주가 상승 기대는 과도하며 장기적 성장 모멘텀은 지나친 낙관론입니다."

    new_history = f"비관론자(Turn {turn}): {clean_output}"
    print(new_history)
    return {
        "debate_history" : [new_history],
        "turn_count" : turn + 1,
        "current_agent" : "pessimist",
        "tool_calls" : response.tool_calls
    }

def summary_node(state: DebateAgentState):
    """
    중재자 에이전트: 토론 내용을 종합하여 최종 결론을 내립니다.
    """
    print("\n😐 [중재자] ------------------")
    ticker = state["ticker"]
    # 프롬프트 로드
    system_prompt = load_prompt("neutral_prompt")
    # 토론 맥락 취합
    history_str = "\n\n".join(state.get("debate_history", []))
    # 입력 메시지 구성
    input_message = {
        "ticker": ticker, 
        "optimist_initial": state.get("optimist_initial", "내용 없음."),
        "pessimist_initial": state.get("pessimist_initial", "내용 없음."),
        "history": history_str
    }
    # 에이전트 실행 (Tool Calling 필요 없으므로 일반 invoke 사용)
    consensus = None
    try:
        consensus = call_kanana_structured(
            system_prompt = system_prompt,
            user_input = input_message,
            output_schema = ConsensusOutput,
            max_new_tokens = Config.KANANA_SUMMARY_MAX_NEW_TOKENS
        )
    # 결과 반환
        final_report = consensus.to_report_text
        
    except Exception as e:
        print(f"❌ 중재자 노드 오류: {e}")
        final_report = "최종 결론 도출에 실패했습니다. 토론 기록을 참고해주세요."

    recommendation = consensus.recommendation if consensus else "보류(파싱 실패)"
    print(f"\n[⚖️ 최종 투자 의견]: {recommendation}")
    print(f"------------------------------------------\n{final_report}")

    return {
        "final_consensus": final_report
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
        f"{'='*50}",
        f" Multi-Agent Investment Analysis Report: {ticker}",
        f" Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*50}",
        "\n[1. Optimist Initial Opinion]", state.get("optimist_initial", ""),
        "\n[2. Pessimist Initial Opinion]", state.get("pessimist_initial", ""),
        "\n[3. Full Debate History]", "\n".join(state.get("debate_history", [])),
        "\n[4. Final Consensus Report]", state.get("final_consensus", "No consensus reached."),
        f"\n{'='*50}",
    ]
    try:
        report_content = "\n".join(full_report)
        with open(debate_path / "full_report.txt", "w", encoding = "utf-8") as f:
            f.write(report_content)
        print(f"[System] '{ticker}' 토론 결과 저장 완료.")
    except Exception as e:
        print(f"[System] 토론 결과 저장 실패: {e}")
    return state    