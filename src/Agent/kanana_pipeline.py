import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from typing import Any, Optional, List
import os
import sys
import time
import json
import re

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
            do_sample = True,
            temperature = 0.5,
            return_full_text = False,
            eos_token_id = tokenizer.eos_token_id
        )
        call_time = time.time() - call_start
        
        # 응답 검증 (비어있는 경우)
        if not response or len(response) == 0:
            print("⚠️ Kanana 파이프라인이 빈 응답을 반환했습니다.")
            return ""
        
        raw = response[0] # 결과를 항상 리스트에 담아서 줌 

        # 파이프라인이 문자열을 직접 반환하는 경우
        if isinstance(raw, str):
            result = raw 
        elif isinstance(raw, dict):
            result = raw.get("generated_text", "")
            # generated_text가 List[Dict]인 경우
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

def call_kanana_structured(system_prompt: str, user_input: dict, output_schema: type, max_new_tokens: int = KANANA_MAX_NEW_TOKENS) -> Any:
    """
    Kanana 모델의 output 형태를 한정(JSON)하여 호출하는 함수
    """
    from pydantic import ValidationError

    schema_prompt = (
        "\n\n[출력 형식]\n"
        "아래 형식을 따르는 **하나의 JSON 객체만** 출력하세요.\n"
        "아래 스키마에 포함되지 않은 다른 필드는 포함해서는 안 됩니다.\n"
        "키 이름과 값만 포함된 JSON 형태로 답변해야 합니다.\n\n"
        f"{output_schema.model_json_schema()}\n\n"
        "[중요]\n"
        "- 반드시 최상위에 실제 필드 값들만 있는 JSON 객체를 출력하세요.\n"
        "- 예: {{\"pros\": \"...\", \"cons\": \"...\", \"conclusion\": \"...\", \"recommendation\": \"...\"}}\n\n"
    )
    full_prompt = system_prompt + schema_prompt
    
    if Config.ENABLE_LOCAL_LOGGING:
        log_agent_action("Structured Output 호출", {
            "output_schema": output_schema.__name__,
            "user_input_keys": list(user_input.keys()),
            "max_new_tokens": max_new_tokens
        })

    response_text = call_kanana(full_prompt, user_input, max_new_tokens = max_new_tokens)

    # JSON 파싱 및 Pydantic 검증
    try:
        codeblock_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL | re.IGNORECASE)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        else:
            json_str = _extract_first_json(response_text)

        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(json_str.strip())
        data = _normalize_recommendation(data)
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

        # 1차 실패 시: 모델이 만든 응답 복구 시도
        repair_prompt = (
            "당신의 작업은 아래 텍스트를 스키마에 맞는 JSON 한 개로 정리하는 것입니다.\n"
            "설명, 마크다운 코드블록, 추가 문장 없이 JSON 객체만 출력하세요.\n"
            "값이 불완전하거나 확실하지 않으면 빈 문자열 또는 빈 리스트를 사용하세요.\n"
            "특히 JSON 문자열이 중간에 끊기지 않도록 각 문자열을 짧고 완결되게 작성하세요.\n\n"
            "[스키마]\n"
            f"{output_schema.model_json_schema()}\n\n"
            "[원본 텍스트]\n"
            f"{response_text}\n"
        )

        repaired_text = call_kanana(repair_prompt, {}, max_new_tokens = max(max_new_tokens, 1536))
        try:
            repaired_data = _extract_first_json(repaired_text)
            if not isinstance(repaired_data, dict):
                raise ValueError("No JSON object extracted from repaired output")
            repaired_result = output_schema(**repaired_data)
            if Config.ENABLE_LOCAL_LOGGING:
                log_agent_action("Structured Output 파싱 성공(복구 패스)", {
                    "schema": output_schema.__name__
                })
            return repaired_result

        except (ValidationError, ValueError) as repair_err:
            print(f"❌ Structured Output 복구 파싱 실패 [{output_schema.__name__}]: {repair_err}")
            print(f"복구 응답: {repaired_text[:300]}")
            if Config.ENABLE_LOCAL_LOGGING:
                from utils.logger import log_error
                log_error(repair_err, f"call_kanana_structured(repair) - Schema: {output_schema.__name__}")
            raise

def _extract_first_json(text: Any) -> dict | None:
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

    for i, ch in enumerate(text[start:], start=start): # 나머지 부분에서
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
        if ch == '{': # 한 칸 더 들어가고
            depth += 1
        elif ch == '}': # 찾으면 stop
            depth -= 1
            if depth == 0: # 깊이가 0이라면 딱 맞춰진 { }을 찾았다는 뜻 -> 그 부분만 자르기
                return text[start:i + 1]

    # 닫는 }를 끝까지 못 찾은 경우 — 잘린 응답이므로 시작부터 끝까지 반환
    return text[start:].strip()

def _normalize_recommendation(data: Any) -> Any:
    """recommendation 필드를 스키마 허용값(매수/매도/보류)으로 정규화."""
    if not isinstance(data, dict):
        return data

    raw = data.get("recommendation")
    if raw is None:
        return data

    text = str(raw).strip()
    if text in {"매수", "매도", "보류"}:
        return data

    if "매수" in text:
        data["recommendation"] = "매수"
    elif "매도" in text:
        data["recommendation"] = "매도"
    elif "보류" in text:
        data["recommendation"] = "보류"

    return data

def extract_pure_text(raw_text: Any) -> str:
    """
    모델이 output 필드 내부에 JSON을 중첩하거나 마크다운 코드블록을 포함한 경우 정리
    -> 순수한 출력 문자열만 반환 (최종 리포트 저장용)
    """
    if not raw_text: return ""
    if '"output":' in raw_text or '"action":' in raw_text:
        import json
        import re
        try:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                json_obj = json.loads(match.group(0))
                return json_obj.get("output", raw_text)
        except:
            match = re.search(r'"output"\s*:\s*"(.*?)"', raw_text, re.DOTALL)
            if match:
                return match.group(1).replace('\\n', '\n')
    
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()
    return clean_text

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
        parsed = _extract_first_json(stripped)
        if isinstance(parsed, dict) and "output" in parsed:
            return str(parsed["output"]).strip()

        # JSON 파싱 실패 시 regex fallback
        match = re.search(
            r'"output"\s*:\s*"((?:[^"\\]|\\.)*)"',
            stripped, re.DOTALL
        )
        if match:
            return match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()

    return stripped
    