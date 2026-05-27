from typing import Dict, Literal, Any, List
import yaml
import json
from datetime import datetime
from src.Agent.states import DebateAgentState
from src.Agent.kanana_pipeline import call_kanana, _extract_first_json, _extract_output_only
from src.Agent.schemas import InitialOutput, DebateOutput
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

def _collect_sources(
    tool_name: str,
    args: dict,
    result: Any,
    news_index: dict,
    filing_index: dict,
) -> List[dict]:
    """tool 호출 결과에서 출처 메타데이터만 추출 (뉴스: 제목/url, 공시: 종류/날짜)"""
    collected: List[dict] = []

    if tool_name == "search_recent_news" and isinstance(result, list):
        for row in result:
            if not isinstance(row, dict):
                continue
            article_id = str(row.get("article_id", ""))
            title = str(row.get("title", "") or "")
            url = str(row.get("html", "") or "")
            if article_id:
                news_index[article_id] = {"title": title, "url": url}
            if title or url:
                collected.append({"source_type": "news", "title": title, "url": url})

    elif tool_name == "search_recent_filings" and isinstance(result, list):
        for row in result:
            if not isinstance(row, dict):
                continue
            parsed_path = str(row.get("parsed_path", "") or "")
            form = str(row.get("form", "") or "")
            filed_date = str(row.get("filed_date", "") or "")
            if parsed_path:
                filing_index[parsed_path] = {"form": form, "filed_date": filed_date}
            if form or filed_date:
                collected.append({"source_type": "filing", "form": form, "filed_date": filed_date})

    elif tool_name == "read_news_content":
        article_id = str(args.get("article_id", ""))
        meta = news_index.get(article_id)
        if meta and (meta.get("title") or meta.get("url")):
            collected.append({
                "source_type": "news",
                "title": meta["title"],
                "url": meta["url"],
            })

    elif tool_name == "read_parsed_filing":
        file_path = str(args.get("file_path", ""))
        meta = filing_index.get(file_path)
        if meta and (meta.get("form") or meta.get("filed_date")):
            collected.append({
                "source_type": "filing",
                "form": meta["form"],
                "filed_date": meta["filed_date"],
            })

    return collected


def unique_sources(sources: List[dict]) -> List[dict]:
    """출처 중복 제거 (뉴스: url, 공시: form+filed_date 기준)"""
    seen: set = set()
    unique: List[dict] = []
    for source in sources:
        if source.get("source_type") == "news":
            key = ("news", source.get("url") or source.get("title"))
        else:
            key = ("filing", source.get("form"), source.get("filed_date"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def format_sources_block(sources: List[dict]) -> str:
    """state에 모인 출처를 보고서용 텍스트로 포맷"""
    news = [s for s in sources if s.get("source_type") == "news"]
    filings = [s for s in sources if s.get("source_type") == "filing"]
    if not news and not filings:
        return ""

    lines = ["## 참고자료"]
    if news:
        lines.append("[뉴스]")
        for item in news:
            title = item.get("title") or "제목 없음"
            url = item.get("url", "")
            lines.append(f"- {title}\n  ({url})" if url else f"- {title}")
    if filings:
        lines.append("[공시]")
        for item in filings:
            form = item.get("form", "")
            filed_date = item.get("filed_date", "")
            lines.append(f"- {form} ({filed_date})")
    return "\n".join(lines)


def _build_agent_output(agent_role: str, text: str, tool_calls: list, sources: list):
    """agent_role에 맞는 Output 객체 생성"""
    unique = unique_sources(sources)
    if agent_role == "initial":
        return InitialOutput(text = text, tool_calls = tool_calls, sources = unique)
    return DebateOutput(text = text, tool_calls = tool_calls, sources = unique)
    

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
            max_steps = 2
            tool_calls = []
            sources = []
            news_index = {}
            filing_index = {}
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
                        sources.extend(_collect_sources(
                            tool_name, args, auto_tool_call_result, news_index, filing_index
                        ))
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
                            sources.extend(_collect_sources(
                                "read_news_content", args, read_result, news_index, filing_index
                            ))
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
                            sources.extend(_collect_sources(
                                "read_parsed_filing", args, read_result, news_index, filing_index
                            ))
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

                today = datetime.now().strftime("%Y년 %m월 %d일")
                
                # 매번 기존 프롬프트에 더해서 넣어주는 내용
                iteration_prompt = (
                    f"{system_prompt}\n\n"
                    "### 최신성 사수: 시점 관리 규칙 ###\n"
                    f"1. [현재 시점]: 지금은 **{today}**입니다.\n"
                    "2. [데이터 필터링]: 2023~2024년 데이터는 '과거 기록'일 뿐입니다. 반드시 **2025년 하반기 이후의 뉴스 및 공시**를 우선적으로 탐색하십시오.\n"
                    "3. [응답 생성 시]: 응답에 2025년 이전의 이야기(2023~2024년)가 포함되면 즉시 **응답 실패**로 간주됩니다.\n\n"
                    
                    "### 데이터 필터링 규칙: 주제 식별 ###\n"
                    f"1. 기사의 주인공 확인: 뉴스나 공시의 수혜자가 반드시 {ticker} 본체여야 합니다.\n"
                    f"2. 간접 호재 배제: 파트너사, 경쟁사, 혹은 산업 전반의 호재를 {ticker}의 직접적인 실적 호재로 둔갑시키지 마십시오.\n"
                    f"3. 질문 던지기: 이 사건이 {ticker}의 재무제표(매출/영업이익)에 '직접적'으로 숫자를 바꿀 수 있는가? 를 자문하고, 아니라면 '간접 참고 자료'로만 분류하십시오.\n"
                    f"4. 기사의 내용이 {ticker}와 직접적인 관련이 없다면, 무시하고 넘어가도 괜찮습니다.\n\n"
                    
                    "### 필수 출력 규칙 ###\n"
                    "1. 추가 정보가 필요하면 반드시 아래 JSON 형식으로 도구를 호출하십시오.\n"
                    '{"action": "tool", "tool_name": "...", "args": {...}}\n'
                    "2. 충분한 정보가 모였다면, JSON 형식을 무시하고 즉시 '최종 분석 리포트' 본문(한글 6-8문장)만 작성하십시오.\n"
                    "3. 최종 답변 시에는 절대로 JSON 포맷을 지키려 애쓰지 말고, 분석 내용만 그대로 출력하십시오.\n\n"
                    f"4. 현재 실행 단계가 많아질수록(최대 {max_steps}단계), 반드시 'final' 답변 생성을 우선시하십시오.\n\n"

                    "### 중복 답변 금지 규칙 ###\n"
                    f"1. 당신의 이전 답변 기록({scratch_text})에 포함된 문장이나 표현을 **그대로 복사하지 마십시오.**\n"
                    "2. 반드시 새로운 근거와 논리를 1개 이상 추가하거나, 이전과 다른 각도에서 반박하십시오.\n"
                    "3. 동일한 논리를 반복하는 것은 **분석 실패**로 간주됩니다.\n\n"

                    "원하는 방향에 따라 아래 두 가지 행동 중 한 가지를 선택하세요.\n"
                    "1) 추가 정보가 필요할 때 (중복 호출 금지):\n"
                    '{"action": "tool", "tool_name": "도구 이름", "args": {"key": "value"}}\n'
                    "2) 수집된 정보로 반박/분석이 가능할 때 (최종 답변):\n"
                    "JSON이 아닌, 일반 문자열 분석 본문만 출력\n\n"

                    "### 체크리스트 ###\n"
                    "기존 조사 기록에 없는 새로운 정보가 더 필요한가요?"
                    "그렇지 않다면 지금 즉시 'final' 액션으로 전환하여 결론을 내십시오."
                    "인사말이나 서론 없이, 응답만을 문장으로 출력하세요."
                )
                
                model_text = call_kanana(iteration_prompt, {}, max_new_tokens = KANANA_MAX_NEW_TOKENS).strip()
                decision = _extract_first_json(model_text)

                # 결과 파싱해오기
                if not isinstance(decision, dict) or "action" not in decision:
                    action = "final"
                    final_output = model_text.strip()
                else:
                    action = str(decision.get("action", "")).lower().strip()
                    final_output = str(decision.get("output", "")).strip()

                if action == "final":
                    if len(final_output) < 30:
                        final_output = model_text.strip()

                    # 역할별 출력값 할당
                    return _build_agent_output(agent_role, final_output, tool_calls, sources)

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
                    sources.extend(_collect_sources(
                        tool_name, args, tool_result, news_index, filing_index
                    ))
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

            return _build_agent_output(agent_role, final_output, tool_calls, sources)

    return _AgentExecutor()

def should_continue(state: DebateAgentState) -> Literal["continue", "stop"]:
    """
    should_continue_node의 결과를 받아서 다음 단계로 라우팅
    """
    if state.get("turn_count", 0) >= state.get("max_turns", 4):
        return "stop"
    decision = state.get("should_continue", "continue")
    return "stop" if decision == "stop" else "continue"


