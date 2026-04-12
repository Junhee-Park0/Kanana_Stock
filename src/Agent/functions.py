from typing import Dict, Literal, Any
import yaml
import json

from src.Agent.states import DebateAgentState
from src.Agent.kanana_pipeline import call_kanana

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

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i, ch in enumerate(text[start:], start = start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return {}

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def _extract_output_only(text: str) -> str:
    """모델 응답에 {"action":"final","output":"..."} 등의 형식이 섞여 있어도 실제 output 문자열만 추출"""
    if not isinstance(text, str):
        return str(text)

    parsed = _extract_first_json_object(text)
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "")).lower().strip()
        if action == "final" and "output" in parsed:
            return str(parsed.get("output", "")).strip()
    return text.strip()


def create_agent(llm, tools, system_prompt):
    """Kanana용 수동 tool-calling 루프 (nodes.py 인터페이스 호환)"""
    del llm  # 현재 create_agent 인터페이스 호환만 유지
    tool_map = {tool.name: tool for tool in tools}

    class _AgentExecutorCompat:
        def invoke(self, payload):
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
            max_steps = 4
            tool_calls = []
            ticker = str(payload.get("ticker", "")).strip().upper()

            # 모델이 툴 호출을 누락하는 경우를 대비해 핵심 컨텍스트를 자동 조회
            auto_tools = []
            if "search_recent_news" in tool_map:
                auto_tools.append(("search_recent_news", {"ticker": ticker, "limit": MAX_NEWS_COUNT}))
            if "search_recent_filings" in tool_map:
                auto_tools.append(("search_recent_filings", {"ticker": ticker, "days": MAX_SEC_DAYS}))

            auto_results = {}
            if ticker and auto_tools:
                for tool_name, args in auto_tools:
                    try:
                        auto_result = tool_map[tool_name].invoke(args)
                        auto_results[tool_name] = auto_result
                        auto_result_text = str(auto_result)
                        if len(auto_result_text) > 3000:
                            auto_result_text = auto_result_text[:3000] + "...(truncated)"
                        result_count = len(auto_result) if isinstance(auto_result, list) else None
                        tool_calls.append({
                            "step": 0,
                            "tool_name": tool_name,
                            "args": args,
                            "result_count": result_count,
                        })
                        print(
                            f"[ToolCall-Auto] tool={tool_name} "
                            f"args={json.dumps(args, ensure_ascii = False)} "
                            f"result_count={result_count if result_count is not None else 'n/a'}"
                        )
                        scratchpad.append(
                            f"[Step 0] Tool `{tool_name}` args={json.dumps(args, ensure_ascii = False)}\n"
                            f"Result: {auto_result_text}"
                        )
                    except Exception as e:
                        scratchpad.append(
                            f"[Step 0] Tool `{tool_name}` failed: {type(e).__name__}: {e}"
                        )

                # 검색만 하고 끝나는 것을 방지하기 위해 본문 read 툴을 자동 호출
                # (뉴스/공시 각각 최대 2개까지)
                news_rows = auto_results.get("search_recent_news", [])
                if isinstance(news_rows, list) and "read_news_content" in tool_map:
                    for row in news_rows[:2]:
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
                                "step": 0,
                                "tool_name": "read_news_content",
                                "args": args,
                                "result_count": None,
                            })
                            print(
                                f"[ToolCall-AutoRead] tool=read_news_content "
                                f"args={json.dumps(args, ensure_ascii = False)}"
                            )
                            scratchpad.append(
                                f"[Step 0] Tool `read_news_content` args={json.dumps(args, ensure_ascii = False)}\n"
                                f"Result: {read_text}"
                            )
                        except Exception as e:
                            scratchpad.append(
                                f"[Step 0] Tool `read_news_content` failed: {type(e).__name__}: {e}"
                            )

                filing_rows = auto_results.get("search_recent_filings", [])
                if isinstance(filing_rows, list) and "read_parsed_filing" in tool_map:
                    for row in filing_rows[:2]:
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
                                "step": 0,
                                "tool_name": "read_parsed_filing",
                                "args": args,
                                "result_count": None,
                            })
                            print(
                                f"[ToolCall-AutoRead] tool=read_parsed_filing "
                                f"args={json.dumps(args, ensure_ascii = False)}"
                            )
                            scratchpad.append(
                                f"[Step 0] Tool `read_parsed_filing` args={json.dumps(args, ensure_ascii = False)}\n"
                                f"Result: {read_text}"
                            )
                        except Exception as e:
                            scratchpad.append(
                                f"[Step 0] Tool `read_parsed_filing` failed: {type(e).__name__}: {e}"
                            )

            for step in range(1, max_steps + 1):
                scratch_text = "\n".join(scratchpad) if scratchpad else "(없음)"
                iteration_prompt = (
                    f"{system_prompt}\n\n"
                    "[도구 사용 규칙]\n"
                    "반드시 아래 JSON 중 하나만 출력하세요.\n"
                    "1) 도구 호출:\n"
                    '{"action":"tool","tool_name":"<tool_name>","args":{"key":"value"}}\n'
                    "2) 최종 답변:\n"
                    '{"action":"final","output":"최종 분석 텍스트"}\n\n'
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
                    print(
                        f"[ToolCall] step={step} tool={tool_name} "
                        f"args={json.dumps(args, ensure_ascii = False)} "
                        f"result_count={result_count if result_count is not None else 'n/a'}"
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
            return {"output": final_output, "tool_calls": tool_calls}

    return _AgentExecutorCompat()

def should_continue(state: DebateAgentState) -> Literal["optimist", "summary"]:
    """
    토론을 계속할지 중재자로 넘어갈지 결정하는 조건부 엣지 함수
    """
    if state["turn_count"] >= state["max_turns"]:
        return "summary"
    return "optimist"


