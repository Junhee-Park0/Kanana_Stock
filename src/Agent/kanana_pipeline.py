import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from typing import Any, Optional, List
import os
import sys
import time

# Config 및 Logger 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config
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
            torch_dtype = torch.float32
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
                temperature = None,
                do_sample = False,
                return_full_text = False,
                eos_token_id = _tokenizer.eos_token_id
            )
            print(f"   ✓ GPU 워밍업 완료 ({time.time() - warmup_start:.1f}초)")
        except Exception as e:
            print(f"   ⚠️ GPU 워밍업 실패 (무시됨): {e}")

    return _pipeline, _tokenizer

def call_kanana(system_prompt: str, user_input: dict, max_new_tokens: int = 512) -> str:
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
            temperature = None,
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
    텍스트에서 첫 번째로 완성된 JSON 객체를 추출한다.
    
    greedy 정규식('{.*}') 대신 중괄호 깊이를 직접 추적하여
    '첫 { ~ 그에 대응하는 }' 구간만 정확히 잘라낸다.
    이렇게 하면 JSON 뒤에 추가 텍스트가 붙어 있어도 안전하다.
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


def call_kanana_structured(system_prompt: str, user_input: dict, output_schema: type, max_new_tokens: int = 512) -> Any:
    """
    Kanana 모델의 output 형태를 한정(JSON)하여 호출하는 함수
    """
    import json
    import re
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
        # 1) ```json ... ``` 코드블록 안의 JSON 우선 탐색 (greedy 매칭으로 변경하여 내부 }에 의한 조기 종료 방지)
        codeblock_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        else:
            # 2) 중괄호 깊이 추적으로 첫 번째 완전한 JSON 객체만 추출
            json_str = _extract_first_json(response_text)

        # raw_decode: JSON 뒤에 여분의 텍스트가 있어도 첫 번째 유효한 JSON만 파싱
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_str.strip())
        result = output_schema(**data)

        if Config.ENABLE_LOCAL_LOGGING:
            log_agent_action("Structured Output 파싱 성공", {
                "schema": output_schema.__name__
            })
        return result

    except (json.JSONDecodeError, ValidationError) as e:
        print(f"❌ Structured Output 파싱 실패 [{output_schema.__name__}]: {e}")
        print(f"원본 응답: {response_text[:300]}")
        
        if Config.ENABLE_LOCAL_LOGGING:
            from utils.logger import log_error
            log_error(e, f"call_kanana_structured - Schema: {output_schema.__name__}")
        raise

# 툴 바인딩을 위한..
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

class ChatKanana(BaseChatModel):
    """LangChain 에이전트와 호환되는 Kanana 커스텀 채팅 모델 클래스"""
    model_name : str = "Kanana-Local"
    max_new_tokens : int = 512

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
        
        response = pipeline(
            hf_messages,
            max_new_tokens = self.max_new_tokens,
            temperature = kwargs.get("temperature", 0.0),
            do_sample = True if kwargs.get("temperature", 0.0) > 0.0 else False,
            eos_token_id = tokenizer.eos_token_id,
            return_full_text = False
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

