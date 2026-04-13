import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from typing import Any, Optional, List
import os
import sys
import time

# Config 및 Logger 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config
KANANA_MAX_NEW_TOKENS = Config.KANANA_MAX_NEW_TOKENS
from utils.logger import logger, log_agent_action

_pipeline = None
_tokenizer = None

def get_kanana_pipeline():
    """Kanana 모델 파이프라인을 받아오는 함수"""
    global _pipeline, _tokenizer

    if _pipeline is None:
        start_time = time.time()
        model_path = Config.KANANA_MODEL_PATH
        
        print("📥 로컬 토크나이저 로드 중...")
        tokenizer_start = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex = True)
        print(f"   ✓ 토크나이저 로드 완료 ({time.time() - tokenizer_start:.1f}초)")

        print("📦 로컬 모델 로드 중...")
        model_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map = "auto",
            torch_dtype = torch.float16
        )
        print(f"   ✓ 로컬 모델 로드 완료 ({time.time() - model_start:.1f}초)")
        
        print("🔧 파이프라인 생성 중...")
        pipeline_start = time.time()
        _pipeline = hf_pipeline(
            "text-generation",
            model = model,
            tokenizer = _tokenizer
        )
        print(f"   ✓ 파이프라인 생성 완료 ({time.time() - pipeline_start:.1f}초)")
        
        total_time = time.time() - start_time
        print(f"✅ Kanana 모델 파이프라인 로드 완료 (총 {total_time:.1f}초)")
        
        # CUDA 커널 워밍업: 첫 번째 실제 추론 전에 더미 호출로 JIT 컴파일 수행
        # 이렇게 하면 첫 질문 처리 시 추가 지연이 없어집니다.
        print("🔥 GPU 워밍업 중... (첫 질문 응답 속도 향상을 위한 사전 작업)")
        warmup_start = time.time()
        try:
            warmup_messages = [
                {"role": "system", "content": "당신은 법률 전문가입니다."},
                {"role": "user", "content": "안녕하세요."}
            ]
            _ = _pipeline(
                warmup_messages,
                max_new_tokens = 10,
                do_sample = False,
                return_full_text = False,
                eos_token_id = _tokenizer.eos_token_id
            )
            print(f"   ✓ GPU 워밍업 완료 ({time.time() - warmup_start:.1f}초)")
        except Exception as e:
            print(f"   ⚠️ GPU 워밍업 실패 (무시됨): {e}")

    return _pipeline, _tokenizer

def call_kanana(system_prompt: str, user_input: dict, max_new_tokens: int = KANANA_MAX_NEW_TOKENS) -> str:
    """
    Kanana 모델을 직접 호출하는 함수
    """
    pipeline, tokenizer = get_kanana_pipeline()

    formatted_system = system_prompt
    for key, value in user_input.items():
        formatted_system = formatted_system.replace(f"{{{key}}}", str(value))
    
    messages = [
        {"role": "system", "content": formatted_system},
        {"role": "user", "content": "위 지시사항에 따라 처리해주세요."}
    ]
    
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("Kanana 호출", {
            "max_new_tokens": max_new_tokens,
            "prompt_length": len(system_prompt),
            "user_input_keys": list(user_input.keys())
        })
    
    try:
        call_start = time.time()
        response = pipeline(
            messages,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            return_full_text = False,
            eos_token_id = tokenizer.eos_token_id
        )
        call_time = time.time() - call_start
        
        # 응답 검증 (비어있는 경우)
        if not response or len(response) == 0:
            print("⚠️ Kanana 파이프라인이 빈 응답을 반환했습니다.")
            return ""
        
        raw = response[0]
        if isinstance(raw, str):
            # 파이프라인이 문자열을 직접 반환하는 경우
            result = raw
        elif isinstance(raw, dict):
            result = raw.get("generated_text", "")
            # 채팅 템플릿 모델: generated_text가 메시지 dict 리스트인 경우
            if isinstance(result, list):
                last = result[-1] if result else ""
                if isinstance(last, dict):
                    result = last.get("content", "")
                else:
                    result = str(last)
        else:
            result = str(raw)

        if not result or result.strip() == "":
            print("⚠️ Kanana가 빈 텍스트를 생성했습니다.")
            print(f"   원본 응답: {response}")
        
        if Config.ENABLE_LOCAL_LOGGING:
            log_agent_action("Kanana 응답 완료", {
                "response_length": len(result),
                "call_time": call_time,
                "has_response": bool(result)
            })
        return result

    except Exception as e:
        import traceback
        print(f"❌ Kanana 모델 호출 중 오류가 발생했습니다: {e}")
        print(f"   상세 오류: {traceback.format_exc()}")
        print(f"   프롬프트 길이: {len(formatted_system)}")
        print(f"   max_new_tokens: {max_new_tokens}")
        if Config.ENABLE_LOCAL_LOGGING:
            from utils.logger import log_error
            log_error(e, "call_kanana")
        raise

def _extract_first_json(text: str) -> str:
    """
    텍스트에서 첫 번째로 완성된 JSON 객체를 추출
    
    정규식('{.*}') 대신 중괄호 깊이를 직접 추적하여
    '첫 { ~ 그에 대응하는 }' 구간만 정확히 잘라내기
    이렇게 하면 JSON 뒤에 추가 텍스트가 붙어 있어도 안전
    """
    start = text.find('{')
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # 닫는 }를 끝까지 못 찾은 경우 — 잘린 응답이므로 시작부터 끝까지 반환
    return text[start:].strip()


def call_kanana_structured(system_prompt: str, user_input: dict, output_schema: type, max_new_tokens: int = KANANA_MAX_NEW_TOKENS) -> Any:
    """
    Kanana 모델의 output 형태를 한정(JSON)하여 호출하는 함수
    """
    from pydantic import ValidationError

    schema_description = (
        "\n\n[출력 형식]\n"
        "아래는 참고용 JSON Schema입니다. 이 스키마를 그대로 출력하지 말고,\n"
        "해당 스키마를 따르는 **하나의 JSON 객체만** 출력하세요.\n"
        "설명(description), properties, type 등의 메타데이터는 출력하지 마세요.\n"
        "키 이름과 값만 포함된 JSON 예시 형태로 답변해야 합니다.\n\n"
        f"{output_schema.model_json_schema()}\n\n"
        "[중요]\n"
        "- 반드시 최상위에 실제 필드 값들만 있는 JSON 객체를 출력하세요.\n"
        "- 예: {{\"enough_context\": \"ENOUGH\", \"reason\": \"...\"}}\n\n"
    )
    full_prompt = system_prompt + schema_description
    
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("Structured Output 호출", {
            "output_schema": output_schema.__name__,
            "user_input_keys": list(user_input.keys()),
            "max_new_tokens": max_new_tokens
        })

    response_text = call_kanana(full_prompt, user_input, max_new_tokens = max_new_tokens)

    # JSON 파싱 및 Pydantic 검증
    try:
        data = _extract_first_json_object(response_text)
        if not data:
            raise ValueError("No JSON object extracted from model output")
        result = output_schema(**data)

        if Config.ENABLE_LOCAL_LOGGING:
            log_agent_action("Structured Output 파싱 성공", {
                "schema": output_schema.__name__
            })
        return result

    except (ValidationError, ValueError) as e:
        print(f"❌ Structured Output 파싱 실패 [{output_schema.__name__}]: {e}")
        print(f"원본 응답: {response_text[:300]}")
        
        if Config.ENABLE_LOCAL_LOGGING:
            from utils.logger import log_error
            log_error(e, f"call_kanana_structured - Schema: {output_schema.__name__}")
        raise

def _fix_json_newlines(json_str: str) -> str:
    """
    모델 출력 JSON에서 문자열 값 안에 들어간 실제 개행문자를
    JSON 이스케이프 시퀀스로 변환하여 파서가 수용할 수 있게 한다.
    """
    result = []
    in_string = False
    escape_next = False
    for ch in json_str:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        if in_string and ch == '\r':
            result.append('\\r')
            continue
        result.append(ch)
    return ''.join(result)


def _regex_extract_fields(text: str) -> dict:
    """
    JSON 파싱이 완전히 실패했을 때 regex로 주요 필드 값만 추출하는 최후 수단 fallback.
    action, output, tool_name, pros, cons, recommendation, conclusion과
    evidence 배열을 추출 시도한다.
    """
    import re, json
    result = {}
    string_fields = [
        "action", "output", "tool_name",
        "pros", "cons", "recommendation", "conclusion",
    ]
    for field in string_fields:
        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text, re.DOTALL
        )
        if match:
            result[field] = match.group(1).replace('\\n', '\n').replace('\\"', '"')

    args_match = re.search(r'"args"\s*:\s*(\{[^{}]*\})', text)
    if args_match:
        try:
            result["args"] = json.loads(args_match.group(1))
        except Exception:
            pass

    # evidence 배열: 배열 깊이 추적으로 정확한 범위 추출 후 파싱
    ev_start = text.find('"evidence"')
    if ev_start != -1:
        bracket_start = text.find('[', ev_start)
        if bracket_start != -1:
            depth = 0
            in_str = False
            esc = False
            ev_end = -1
            for idx, ch in enumerate(text[bracket_start:], start=bracket_start):
                if esc:
                    esc = False
                    continue
                if ch == '\\' and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        ev_end = idx
                        break
            if ev_end != -1:
                evidence_str = text[bracket_start:ev_end + 1]

                # 1차: 직접 파싱
                parsed_ev = None
                for candidate in [evidence_str, _fix_json_newlines(evidence_str)]:
                    try:
                        parsed_ev = json.loads(candidate)
                        break
                    except Exception:
                        pass

                if parsed_ev is not None:
                    result["evidence"] = parsed_ev
                else:
                    # 2차: 항목별 regex 추출 (배열 전체 파싱 실패 시)
                    items = []
                    for m in re.finditer(
                        r'"source"\s*:\s*"(news|filing)"'
                        r'.*?"source_id"\s*:\s*"([^"]*)"'
                        r'.*?"summary"\s*:\s*"((?:[^"\\]|\\.)*)"',
                        evidence_str, re.DOTALL
                    ):
                        items.append({
                            "source": m.group(1),
                            "source_id": m.group(2),
                            "summary": m.group(3).replace('\\n', '\n'),
                        })
                    if items:
                        result["evidence"] = items

    return result


def _extract_first_json_object(text: str) -> dict:
    """
    텍스트에서 첫 번째 완성된 JSON 객체를 추출하고 dict로 파싱하여 반환
    3단계 fallback:
      1차) 직접 json.JSONDecoder 파싱
      2차) 문자열 내부 개행 이스케이프 후 재파싱 (_fix_json_newlines)
      3차) regex로 주요 필드만 추출 (_regex_extract_fields)
    모든 단계 실패 시 빈 dict 반환.
    """
    import json
    from utils.logger import log_json_parse_warning

    json_str = _extract_first_json(text)
    if not json_str:
        return {}

    # 1차: 직접 파싱
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_str.strip())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    # 2차: 문자열 내 개행 이스케이프 후 재파싱
    try:
        fixed = _fix_json_newlines(json_str)
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(fixed.strip())
        log_json_parse_warning("_extract_first_json_object", json_str[:200], fallback_used="fix_newlines")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    # 3차: regex 필드 추출
    result = _regex_extract_fields(text)
    if result:
        log_json_parse_warning("_extract_first_json_object", json_str[:200], fallback_used="regex_extract")
        return result

    log_json_parse_warning("_extract_first_json_object", json_str[:200], fallback_used="all_failed→{}")
    return {}


def _extract_output_only(text: str) -> str:
    """
    모델이 output 필드 내부에 JSON을 중첩하거나 마크다운 코드블록을 포함한 경우 정리
    순수한 출력 문자열만 반환한다.
    fallback: JSON 파싱 실패 시 regex로 "output" 필드 값 직접 추출.
    """
    import re
    if not text or not text.strip():
        return text

    stripped = text.strip()

    # 마크다운 코드블록 제거
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        inner_lines = lines[1:-1] if len(lines) > 2 else lines[1:]
        stripped = "\n".join(inner_lines).strip()

    if stripped.startswith("{"):
        # JSON 파싱으로 output 필드 추출
        parsed = _extract_first_json_object(stripped)
        if "output" in parsed:
            return str(parsed["output"]).strip()

        # JSON 파싱 실패 시 regex fallback
        match = re.search(
            r'"output"\s*:\s*"((?:[^"\\]|\\.)*)"',
            stripped, re.DOTALL
        )
        if match:
            return match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()

    return stripped


# 툴 바인딩을 위한..
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

class ChatKanana(BaseChatModel):
    """LangChain 에이전트와 호환되는 Kanana 커스텀 채팅 모델 클래스 (Tool-Calling 지원 x)"""
    model_name : str = "Kanana-Local"
    max_new_tokens : int = KANANA_MAX_NEW_TOKENS

    def bind_tools(self, tools, **kwargs):
        """
        LangGraph create_react_agent 호환을 위한 최소 구현.
        Kanana는 네이티브 tool-calling 포맷을 지원하지 않으므로
        현재 모델 인스턴스를 그대로 반환한다.
        """
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        """LangChain에서 invoke 호출 시 내부적으로 실행되는 메서드"""
        pipeline, tokenizer = get_kanana_pipeline()

        hf_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                hf_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                hf_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                hf_messages.append({"role": "assistant", "content": message.content})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        
        temperature = kwargs.get("temperature")
        do_sample = isinstance(temperature, (int, float)) and temperature > 0

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "return_full_text": False,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)

        response = pipeline(
            hf_messages,
            **generation_kwargs
        )

        generated_text = ""
        if response and isinstance(response[0], dict):
            generated_text = response[0].get("generated_text", "")
            if isinstance(generated_text, list):
                generated_text = generated_text[-1].get("content", "")

        message = AIMessage(content = generated_text)
        generation = ChatGeneration(message = message)
        return ChatResult(generations = [generation])

    @property
    def _llm_type(self) -> str:
        return "kanana-local-chat"

