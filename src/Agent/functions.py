from typing import Dict, Literal, Any
import yaml
import json

from src.Agent.states import DebateAgentState
from src.Agent.kanana_pipeline import call_kanana, _extract_first_json, _extract_output_only
from src.Agent.schemas import EvidenceItem, InitialOutput, DebateOutput
from utils.logger import log_tool_call

from config import Config
MAX_NEWS_COUNT = Config.MAX_NEWS_COUNT
MAX_SEC_DAYS = Config.MAX_SEC_DAYS
KANANA_MAX_NEW_TOKENS = Config.KANANA_MAX_NEW_TOKENS

def load_prompt(prompt_name: str, **kwargs) -> str:
    """
    prompts.yaml에서 프롬프트를 로드하여, 문자열로 반환합니다.
    (**kwargs : 프롬프트에서 비어있는 부분(ex. ticker, history 등)을 채워주기 위함)
    """
    with open(f"src/Agent/prompts.yaml", "r", encoding = "utf-8") as f:
        prompts = yaml.safe_load(f)
        prompt = prompts.get(prompt_name, {})

    if not isinstance(prompt, dict):
        raise ValueError(f"Prompt '{prompt_name}' must be a mapping object.")

    template = f"{prompt['role']}\n{prompt['instructions']}"
    if not kwargs:
        return template

    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_key = e.args[0]
        raise KeyError(
            f"Prompt '{prompt_name}' requires variable '{missing_key}' but it was not provided."
        ) from e

def create_agent(tools, system_prompt, agent_role: Literal["initial", "debate"] = "initial"):
    """
    Kanana용 수동 Tool-Calling (직접 llm을 불러와 Tool과 연결)
    agent_role에 따라 InitialOutput 또는 DebateOutput 반환
    """
    tool_map = {tool.name: tool for tool in tools} # 각 노드에서 건네줌

    class _AgentExecutor:
        def invoke(self, payload): # payload : {ticker: str, input: str, chat_history: list, opponent_text: str}
            input_text = str(payload.get("input", "")).strip()
            chat_history = payload.get("chat_history", [])
            ticker = str(payload.get("ticker", "")).strip().upper()
            if agent_role == "debate":
                opponent_text = str(payload.get("opponent_text", "")).strip() 
            else:
                opponent_text = ""

            tool_specs = "\n".join(
                f"- {name}: {getattr(tool, 'description', '').strip()}"
                for name, tool in tool_map.items()
            )
            scratchpad = []
            final_output = ""
            raw_evidence = []
            max_steps = 2
            tool_calls = []
            ticker = str(payload.get("ticker", "")).strip().upper()

            # 모델이 툴 호출을 누락하는 경우를 대비해 자동으로 Tool 호출
            auto_tool_call = []
            if "search_recent_news" in tool_map:
                auto_tool_call.append(("search_recent_news", {"ticker": ticker, "limit": MAX_NEWS_COUNT}))
            if "search_recent_filings" in tool_map:
                auto_tool_call.append(("search_recent_filings", {"ticker": ticker, "days": MAX_SEC_DAYS}))

            auto_tool_call_results = {}
            if ticker and auto_tool_call:
                for tool_name, args in auto_tool_call:
                    try:
                        auto_tool_call_result = tool_map[tool_name].invoke(args) # news/filings 검색 결과
                        auto_tool_call_results[tool_name] = auto_tool_call_result
                        auto_tool_call_result_text = str(auto_tool_call_result)
                        if len(auto_tool_call_result_text) > 3000:
                            auto_tool_call_result_text = auto_tool_call_result_text[:3000] + "...(truncated)"
                        result_count = len(auto_tool_call_result) if isinstance(auto_tool_call_result, list) else None # 검색된 news/filings 개수
                        # Tool Call 기록에 추가
                        tool_calls.append({
                            "step": 0, # 0 = 자동 Tool Call 
                            "tool_name": tool_name,
                            "args": args,
                            "result_count": result_count,
                        })
                        log_tool_call(
                            step = 0,
                            tool_name = tool_name,
                            args = args,
                            result_count = result_count
                        )
                        scratchpad.append(
                            f"[Step 0] Tool `{tool_name}` args = {json.dumps(args, ensure_ascii = False)}\n"
                            f"Result: {auto_tool_call_result_text}"
                        )
                    except Exception as e:
                        scratchpad.append(
                            f"[Step 0] Tool `{tool_name}` failed: {type(e).__name__}: {e}"
                        )

                # 검색만 하고 끝나는 것을 방지하기 위해 본문 read Tool을 자동 호출
                # read_news_content
                news_rows = auto_tool_call_results.get("search_recent_news", [])
                if isinstance(news_rows, list) and "read_news_content" in tool_map:
                    for row in news_rows[:10]: # 최대 10개까지 읽기
                        article_id = row.get("article_id") if isinstance(row, dict) else None
                        if article_id is None:
                            continue
                        args = {"article_id": str(article_id)}
                        try:
                            read_result = tool_map["read_news_content"].invoke(args)
                            read_text = str(read_result)
                            if len(read_text) > 3000:
                                read_text = read_text[:3000] + "...(truncated)"
                            tool_calls.append({
                                "step": 0, # 0 = 자동 Tool Call
                                "tool_name": "read_news_content",
                                "args": args,
                                "result_count": None, # read tool은 결과 개수가 의미 x
                            })
                            log_tool_call(
                                step = 0,
                                tool_name = "read_news_content",
                                args = args,
                                result_count = None
                            )
                            scratchpad.append(
                                f"[Step 0] Tool `read_news_content` args = {json.dumps(args, ensure_ascii = False)}\n"
                                f"Result: {read_text}"
                            )
                        except Exception as e:
                            scratchpad.append(
                                f"[Step 0] Tool `read_news_content` failed: {type(e).__name__}: {e}"
                            )
                # read_parsed_filing
                filing_rows = auto_tool_call_results.get("search_recent_filings", [])
                if isinstance(filing_rows, list) and "read_parsed_filing" in tool_map:
                    for row in filing_rows[:5]: # 최대 5개까지 읽기
                        parsed_path = row.get("parsed_path") if isinstance(row, dict) else None
                        if not parsed_path:
                            continue
                        args = {"file_path": str(parsed_path)}
                        try:
                            read_result = tool_map["read_parsed_filing"].invoke(args)
                            read_text = str(read_result)
                            if len(read_text) > 3000:
                                read_text = read_text[:3000] + "...(truncated)"
                            tool_calls.append({
                                "step": 0, # 0 = 자동 Tool Call
                                "tool_name": "read_parsed_filing",
                                "args": args,
                                "result_count": None, # read tool은 결과 개수가 의미 x
                            })
                            log_tool_call(
                                step = 0,
                                tool_name = "read_parsed_filing",
                                args = args,
                                result_count = None
                            )
                            scratchpad.append(
                                f"[Step 0] Tool `read_parsed_filing` args = {json.dumps(args, ensure_ascii = False)}\n"
                                f"Result: {read_text}"
                            )
                        except Exception as e:
                            scratchpad.append(
                                f"[Step 0] Tool `read_parsed_filing` failed: {type(e).__name__}: {e}"
                            )

            # 출력 스키마 설정 (예시) - 모델은 evidence 형식을 여기서 확인
            if agent_role == "initial":
                schema_json = "{'action': 'final', 'output': '분석 결과', 'evidence': [{{'source': 'news 또는 filing', 'source_id': 'ID', 'summary': '요약'}}]}"
            else:
                schema_json = "{'action': 'final', 'output': '상대 의견 반박 내용', 'opponent_text': '상대방 의견 요약', 'evidence': [{{'source': 'news 또는 filing', 'source_id': 'ID', 'summary': '핵심 사실 한 줄'}}]}"

            # 매 step마다 Tool Call 결과 기록, 프롬프트에 주입
            for step in range(1, max_steps + 1): 
                scratch_text = "\n".join(scratchpad) if scratchpad else "(없음)" # Tool Call 중간 작업 기록
                
                # 매번 기존 프롬프트에 더해서 넣어주는 내용
                iteration_prompt = (
                    f"{system_prompt}\n\n"
                    "### 필수 출력 규칙 ###\n"
                    "1. 오직 하나의 유효한 JSON 객체만을 출력해야 합니다. 서론, 부연 설명, 마크다운 코드는 절대 금지합니다.\n"
                    "2. [중요] 동일한 파일이나 뉴스를 반복해서 읽지 마십시오. 이미 '조사 기록'에 있는 ID는 다시 호출할 수 없습니다.\n"
                    "3. [종료 조건] 조사 기록에 충분한 근거(수치, 사실)가 모였다면, 추가 도구 호출 없이 즉시 'final'을 선택하십시오.\n"
                    f"4. 현재 실행 단계가 많아질수록(최대 {max_steps}단계), 반드시 'final' 답변 생성을 우선시하십시오.\n\n"

                    "### 중복 답변 금지 규칙 ###\n"
                    f"1. 당신의 이전 답변 기록({scratch_text})에 포함된 문장이나 표현을 **그대로 복사하지 마십시오.**\n"
                    "2. 반드시 새로운 근거와 논리를 1개 이상 추가하거나, 이전과 다른 각도에서 반박하십시오.\n"
                    "3. 동일한 논리를 반복하는 것은 **분석 실패**로 간주됩니다.\n\n"

                    "원하는 방향에 따라 아래 두 가지 행동 중 한 가지를 선택하세요.\n"
                    "1) 추가 정보가 필요할 때 (중복 호출 금지):\n"
                    '{"action": "tool", "tool_name": "도구 이름", "args": {"key": "value"}}\n'
                    "2) 수집된 정보로 반박/분석이 가능할 때 (최종 답변):\n"
                    f"{schema_json}\n\n"

                    "### 현재 상황 데이터 ###\n"
                    f"- 상대방의 의견:\n{opponent_text}\n\n"
                    f"- 현재까지의 조사 기록:\n{scratch_text}\n"

                    "### 체크리스트 ###\n"
                    "기존 조사 기록에 없는 새로운 정보가 더 필요한가요?"
                    "그렇지 않다면 지금 즉시 'final' 액션으로 전환하여 결론을 내십시오."
                    "꼭 JSON으로만 응답하세요."
                )
                
                model_text = call_kanana(iteration_prompt, {}, max_new_tokens = KANANA_MAX_NEW_TOKENS).strip()
                decision = _extract_first_json(model_text)

                # json 파싱 실패 시
                if not decision:
                    decision = {"action": "final", "output": model_text}

                action = str(decision.get("action", "")).lower().strip()

                if action == "final":
                    # output 부분만 추출하기
                    final_output = _extract_output_only(model_text)
                    json_internal_output = str(decision.get("output", "")).strip()
                    if len(final_output) < 20 and len(json_internal_output) > len(final_output):
                        final_output = json_internal_output
                    # evidence 처리
                    raw_evidence = decision.get("evidence", [])
                    evidence_items = []
                    if isinstance(raw_evidence, list):
                        for evidence in raw_evidence:
                            try:
                                if isinstance(evidence, dict):
                                    evidence["source_id"] = str(evidence.get("source_id", ""))
                                    evidence_items.append(EvidenceItem(**evidence))
                            except: 
                                continue
                    # 역할별 evidence 출력값 할당
                    if agent_role == "initial":
                        return InitialOutput(
                            text = final_output,
                            evidence = evidence_items,
                            tool_calls = tool_calls # 말 안 들으면 빼버리자.. 
                        )
                    
                    elif agent_role == "debate":
                        # 만약 상대의 발언을 요약했다면.. 
                        summarized_opponent = str(decision.get("opponent_text", opponent_text)).strip()

                        return DebateOutput(
                            text = final_output,
                            opponent_text = summarized_opponent, # 말 안 들으면 빼버리자.. 
                            evidence = evidence_items,
                            tool_calls = tool_calls # 말 안 들으면 빼버리자.. 
                        )
                    break

                if action != "tool":
                    final_output = model_text
                    break

                tool_name = str(decision.get("tool_name", "")).strip()
                args = decision.get("args", {})
                if tool_name not in tool_map:
                    scratchpad.append(f"[Step {step}] Unknown tool: {tool_name}")
                    continue

                if not isinstance(args, dict):
                    args = {}

                # Agent의 Tool Call
                try:
                    tool_result = tool_map[tool_name].invoke(args)
                    tool_result_text = str(tool_result)
                    if len(tool_result_text) > 3000:
                        tool_result_text = tool_result_text[:3000] + "...(truncated)"
                    result_count = len(tool_result) if isinstance(tool_result, list) else None
                    tool_calls.append({
                        "step": step,
                        "tool_name": tool_name,
                        "args": args,
                        "result_count": result_count,
                    })
                    log_tool_call(
                        step = step,
                        tool_name = tool_name,
                        args = args,
                        result_count = result_count
                    )
                    scratchpad.append(
                        f"[Step {step}] Tool `{tool_name}` args = {json.dumps(args, ensure_ascii = False)}\n"
                        f"Result: {tool_result_text}"
                    )
                except Exception as e:
                    scratchpad.append(
                        f"[Step {step}] Tool `{tool_name}` failed: {type(e).__name__}: {e}"
                    )

            if not final_output:
                final_output = (
                    "도구 호출 기반 분석을 완료하지 못했습니다. "
                    "입력값 또는 데이터 소스를 확인한 뒤 다시 시도해주세요."
                )
            final_output = _extract_output_only(final_output)

            evidence_items = []
            if isinstance(raw_evidence, list):
                for evidence in raw_evidence:
                    try:
                        evidence_items.append(EvidenceItem(**evidence))
                    except Exception:
                        pass

            if agent_role == "initial":
                return InitialOutput(
                text = final_output,
                    evidence = evidence_items,
                    tool_calls = tool_calls
                )

            elif agent_role == "debate":
                return DebateOutput(
                    text = final_output,
                    opponent_text = opponent_text,
                    evidence = evidence_items,
                    tool_calls = tool_calls
                )

    return _AgentExecutor()

def should_continue(state: DebateAgentState) -> Literal["optimist", "summary"]:
    """
    토론을 계속할지 중재자로 넘어갈지 결정하는 조건부 엣지 함수
    """
    if state["turn_count"] >= state["max_turns"]:
        return "summary"
    return "optimist"


