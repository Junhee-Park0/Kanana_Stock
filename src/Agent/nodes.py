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
    print(f"\n🙂[낙관론자] 초기 의견 도출 중...")
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
    print(f"\n[낙관론자 답변]:\n{clean_output}")
    print(type(clean_output))
    print(f"[참고한 근거 개수]: {len(response.evidence)}개")
    return {
        "optimist_initial" : clean_output,
        "optimist_evidence" : response.evidence,
        "tool_calls" : response.tool_calls
    }

def pessimistic_initial_node(state : DebateAgentState):
    """
    비관론자 에이전트: 부정적인 관점에서 시장을 분석하고 의견을 제시합니다.
    """
    ticker = state["ticker"]
    print(f"\n☹️[비관론자] 초기 의견 도출 중...")
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
    print(f"\n[비관론자 답변]:\n{clean_output}")
    print(type(clean_output))
    print(f"[참고한 근거 개수]: {len(response.evidence)}개")
    return {
        "pessimist_initial" : clean_output,
        "pessimist_evidence" : response.evidence,
        "tool_calls" : response.tool_calls
    }

def optimistic_debate_node(state : DebateAgentState):
    """
    낙관론자 토론 진행 중 : 상대의 논리를 반박하고 긍정적인 근거를 보강
    """
    turn = state.get("turn_count", 0) 
    ticker = state["ticker"]
    print(f"\n🙂[낙관론자 (Turn: {turn})] ------------------")  
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("optimist_debate_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "debate")
    # 토론 맥락 구성
    opponent_initial = state.get("pessimist_initial", "아직 의견이 없습니다.")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    # 입력 메시지 구성 ("반박")
    input_message = (
        f"대상 종목: {ticker}\n\n"
        f"[상대방의 초기 의견]\n{opponent_initial}\n\n"
        f"[지난 토론 기록]\n{history_str}\n\n"
        "### 규칙: 차별화된 반박 수행 ###\n"
        f"당신의 이전 답변인 위 [지난 토론 기록]에 포함된 문장을 그대로 사용하는 것은 엄격히 금지됩니다."
        "반드시 새로운 근거와 논리를 1개 이상 추가하거나, 이전과 다른 각도에서 반박하십시오."
        "새로운 뉴스 ID나 공시 지표를 활용하여 논리를 보강하십시오."
    )
    # 에이전트 실행
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": [],
        "opponent_text": opponent_initial
    })
    print(f"[낙관론자 Turn {turn}] 분석 완료 (도구 사용: {len(response.tool_calls)}회)")
    print(f"[상대 의견 요약]: {response.opponent_text[:50]}...")
    # 결과
    clean_output = extract_pure_text(response.text)
    new_history = f"낙관론자(Turn {turn}): {clean_output}"
    print(type(new_history))
    print(new_history)
    return {
        "debate_history" : [new_history],
        "turn_count": turn + 1,
        "current_agent": "optimist",
        "optimist_evidence" : response.evidence,
        "tool_calls" : response.tool_calls
    }

def pessimistic_debate_node(state : DebateAgentState):
    """
    비관론자 토론 진행 중 
    """
    turn = state.get("turn_count", 0) 
    ticker = state["ticker"]
    print(f"\n☹️[비관론자 (Turn: {turn})] ------------------")
    tools = [search_recent_news, search_recent_filings, read_news_content, read_parsed_filing]
    # 프롬프트 로드
    system_prompt = load_prompt("pessimist_debate_prompt")
    # 에이전트 실행기 생성 (Tool-Calling 지원 o)
    agent_executor = create_agent(tools, system_prompt, agent_role = "debate")
    # 토론 맥락 구성
    opponent_initial = state.get("optimist_initial", "아직 의견이 없습니다.")
    history_list = state.get("debate_history", [])
    history_str = "\n".join(history_list) if history_list else "없음 (첫 번째 반박입니다.)"
    # 입력 메시지 구성 ("반박")
    input_message = (
        f"대상 종목: {ticker}\n\n"
        f"[상대방의 초기 의견]\n{opponent_initial}\n\n"
        f"[지난 토론 기록]\n{history_str}\n\n"
        "### 규칙: 차별화된 반박 수행 ###\n"
        f"당신의 이전 답변인 위 [지난 토론 기록]에 포함된 문장을 그대로 사용하는 것은 엄격히 금지됩니다."
        "반드시 새로운 근거와 논리를 1개 이상 추가하거나, 이전과 다른 각도에서 반박하십시오."
        "새로운 뉴스 ID나 공시 지표를 활용하여 논리를 보강하십시오."
        "위 낙관적인 의견의 허점을 찾아내고, 도구를 사용해 이를 반박할 부정적인 지표나 뉴스를 제시하십시오."
        "그 후, 수치적 근거를 바탕으로 반박 논리를 7문장 내외로 작성해주세요."
    )
    # 에이전트 실행 
    response = agent_executor.invoke({
        "ticker": ticker,
        "input": input_message,
        "chat_history": [],
        "opponent_text": opponent_initial
    })
    print(f"[비관론자 Turn {turn}] 분석 완료 (도구 사용: {len(response.tool_calls)}회)")
    print(f"[상대 의견 요약]: {response.opponent_text[:50]}...")
    # 결과
    clean_output = extract_pure_text(response.text)
    new_history = f"비관론자(Turn {turn}): {clean_output}"
    print(type(new_history))
    print(new_history)
    return {
        "debate_history" : [new_history],
        "turn_count" : turn + 1,
        "current_agent" : "pessimist",
        "pessimist_evidence" : response.evidence,
        "tool_calls" : response.tool_calls
    }

def summary_node(state: DebateAgentState):
    """
    중재자 에이전트: 토론 내용을 종합하여 최종 결론을 내립니다.
    """
    print("\n😐[중재자] ------------------")
    ticker = state["ticker"]
    # 프롬프트 로드
    system_prompt = load_prompt("neutral_prompt")
    # 토론 맥락 취합
    all_evidence = state.get("optimist_evidence", []) + state.get("pessimist_evidence", [])
    evidence_context = "\n".join([f"- {evidence.summary} (출처: {evidence.source_id})" for evidence in all_evidence])
    history_str = "\n\n".join(state.get("debate_history", []))
    # 입력 메시지 구성
    input_message = {
        "ticker": ticker, 
        "optimist_initial": state.get("optimist_initial", "내용 없음."),
        "pessimist_initial": state.get("pessimist_initial", "내용 없음."),
        "debate_history": history_str,
        "collected_evidence": evidence_context
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
    # 수집된 evidence 별도 저장
    all_evidence = state.get("optimist_evidence", []) + state.get("pessimist_evidence", [])
    evidence_report = ["\n[Collected Evidence List]"]
    for evidence in all_evidence:
        evidence_report.append(evidence.to_text)
    try:
        report_content = "\n".join(full_report + evidence_report)
        with open(debate_path / "full_report.txt", "w", encoding = "utf-8") as f:
            f.write(report_content)
        print(f"[System] '{ticker}' 토론 결과 저장 완료.")
    except Exception as e:
        print(f"[System] 토론 결과 저장 실패: {e}")
    return state    