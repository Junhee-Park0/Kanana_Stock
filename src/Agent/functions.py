from typing import Dict, Literal, Any
import yaml
import json
import re

from src.Agent.states import DebateAgentState
from src.Agent.kanana_pipeline import call_kanana

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
                model_text = call_kanana(iteration_prompt, {}, max_new_tokens = 256).strip()
                decision = _extract_first_json_object(model_text)

                if not decision:
                    final_output = model_text
                    break

                action = str(decision.get("action", "")).lower().strip()
                if action == "final":
                    final_output = str(decision.get("output", "")).strip()
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
            return {"output": final_output}

    return _AgentExecutorCompat()

def should_continue(state: DebateAgentState) -> Literal["optimist", "summary"]:
    """
    토론을 계속할지 중재자로 넘어갈지 결정하는 조건부 엣지 함수
    """
    if state["turn_count"] >= state["max_turns"]:
        return "summary"
    return "optimist"


