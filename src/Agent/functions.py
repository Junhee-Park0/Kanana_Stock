from typing import Dict, Literal, Any
import yaml
import json

from src.Agent.states import DebateAgentState
from src.Agent.kanana_pipeline import call_kanana, _extract_first_json_object, _extract_output_only
from src.Agent.schemas import EvidenceItem, InitialOutput
from utils.logger import log_tool_call

from config import Config
MAX_NEWS_COUNT = Config.MAX_NEWS_COUNT
MAX_SEC_DAYS = Config.MAX_SEC_DAYS
KANANA_MAX_NEW_TOKENS = Config.KANANA_MAX_NEW_TOKENS

def load_prompt(prompt_name: str, **kwargs) -> str:
    """
    prompts.yaml에서 프롬프트를 로드하여, 문자열로 반환합니다.
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

def create_agent(tools, system_prompt):
    """Kanana용 수동 Tool-Calling (직접 llm을 불러와 Tool과 연결)"""
    tool_map = {tool.name: tool for tool in tools} # 각 노드에서 건네줌

    class _AgentExecutor:
        def invoke(self, payload): # payload : {ticker: str, input: str, chat_history: list}
            input_text = str(payload.get("input", "")).strip()
            chat_history = payload.get("chat_history", [])
            history_lines = []
            for item in chat_history:
                if isinstance(item, dict):
                    role = str(item.get("role", "unknown"))
                    content = str(item.get("content", ""))
                    history_lines.append(f"[{role}] {content}")
                else:
                    history_lines.append(str(item))
            history_text = "\n".join(history_lines) if history_lines else "(없음)"

            tool_specs = "\n".join(
                f"- {name}: {getattr(tool, 'description', '').strip()}"
                for name, tool in tool_map.items()
            )
            scratchpad = []
            final_output = ""
            raw_evidence = []
            max_steps = 4
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
                    for row in news_rows[:5]: # 최대 5개까지 읽기
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

            # 매 step마다 Tool Call 결과 기록, 프롬프트에 주입
            for step in range(1, max_steps + 1): 
                scratch_text = "\n".join(scratchpad) if scratchpad else "(없음)" # Tool Call 중간 작업 기록
                iteration_prompt = (
                    f"{system_prompt}\n\n"
                    "[도구 사용 규칙]\n"
                    "반드시 아래 JSON 중 하나만 출력하세요.\n"
                    "1) 도구 호출:\n"
                    '{"action": "tool", "tool_name": "<tool_name>", "args": {"key":"value"}}\n'
                    "2) 최종 답변:\n"
                    '{"action": "final", "output": "최종 분석 텍스트", "evidence": [{"source": "news", "source_id": "기사ID", "summary": "핵심 사실 한 줄"}, {"source": "filing", "source_id": "공시ID", "summary": "핵심 사실 한 줄"}]}\n\n'
                    "설명 문장, 코드블록, 마크다운 없이 JSON 객체 하나만 출력하세요.\n\n"
                    f"[사용 가능한 도구]\n{tool_specs}\n\n"
                    f"[이전 대화]\n{history_text}\n\n"
                    f"[사용자 요청]\n{input_text}\n\n"
                    f"[중간 작업 기록]\n{scratch_text}\n"
                )
                model_text = call_kanana(iteration_prompt, {}, max_new_tokens = KANANA_MAX_NEW_TOKENS).strip()
                decision = _extract_first_json_object(model_text)

                if not decision:
                    final_output = model_text
                    break

                action = str(decision.get("action", "")).lower().strip()
                if action == "final":
                    if len(tool_calls) == 0:
                        scratchpad.append(
                            f"[Step {step}] 최종 답변이 조기에 생성되어 도구 호출을 재시도합니다."
                        )
                        continue
                    final_output = _extract_output_only(str(decision.get("output", "")))
                    raw_evidence = decision.get("evidence", [])
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
                        f"[Step {step}] Tool `{tool_name}` args={json.dumps(args, ensure_ascii = False)}\n"
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
                for item in raw_evidence:
                    try:
                        evidence_items.append(EvidenceItem(
                            ticker = ticker,
                            source = item.get("source"),
                            source_id = str(item.get("source_id", "")),
                            summary = str(item.get("summary", "")),
                        ))
                    except Exception:
                        pass

            return InitialOutput(
                ticker = ticker,
                text = final_output,
                evidence = evidence_items,
                tool_calls = tool_calls,
            )

    return _AgentExecutor()

def should_continue(state: DebateAgentState) -> Literal["optimist", "summary"]:
    """
    토론을 계속할지 중재자로 넘어갈지 결정하는 조건부 엣지 함수
    """
    if state["turn_count"] >= state["max_turns"]:
        return "summary"
    return "optimist"


