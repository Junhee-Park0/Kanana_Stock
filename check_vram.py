import torch
import time
import psutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

from config import Config

def trace_memory(label=""):
    """현재 시스템 RAM과 GPU VRAM 사용량을 출력하는 함수"""
    print(f"\n--- [{label}] Memory Trace ---")
    
    # 1. 시스템 RAM 확인
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    print(f"[RAM] 현재 프로세스 점유: {ram_gb:.2f} GB")

    # 2. GPU VRAM 확인
    if torch.cuda.is_available():
        # 실제 할당된 양
        allocated = torch.cuda.memory_allocated() / 1024**3
        # PyTorch가 예약(점유)한 총량
        reserved = torch.cuda.memory_reserved() / 1024**3
        # 장치 전체의 가용 메모리 (nvidia-smi 기준과 유사)
        stats = torch.cuda.get_device_properties(0)
        total_vram = stats.total_memory / 1024**3
        
        print(f"[GPU] Allocated (순수 가중치): {allocated:.2f} GB")
        print(f"[GPU] Reserved (실제 점유): {reserved:.2f} GB")
        print(f"[GPU] 전체 VRAM 중 사용률: {(reserved/total_vram)*100:.1f}% ({total_vram:.1f}GB 기준)")
    else:
        print("[GPU] CUDA를 사용할 수 없습니다. (CPU 모드)")

def load_kanana_safely(model_path):
    trace_memory("시작 전")

    # 1. 토크나이저 로드
    print("\n📥 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    trace_memory("토크나이저 로드 후")

    # 2. 모델 로드
    print("\n📦 Kanana 모델 로드 중 (강제 GPU 할당 모드)...")
    # device_map="auto" 대신 명시적으로 cuda:0을 지정하면 0.00GB 문제를 방지할 수 있습니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True # 로드 시 RAM 폭주 방지
    )
    trace_memory("모델 로드 직후")

    # 3. 파이프라인 생성
    print("\n🔧 파이프라인 생성 중...")
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    trace_memory("파이프라인 생성 후")

    # 4. 워밍업 (실제 추론 시 메모리 변화 확인)
    print("\n🔥 워밍업 중 (KV Cache 할당 확인)...")
    warmup_msg = [{"role": "user", "content": "Hello"}]
    _ = pipe(warmup_msg, max_new_tokens=20)
    trace_memory("워밍업 완료 후")

    return pipe

if __name__ == "__main__":
    load_kanana_safely(Config.KANANA_MODEL_PATH)