"""
Stock Agent FastAPI 서버

실행 방법:
    cd Kanana_Stock
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

엔드포인트:
    GET  /                      - API 정보
    GET  /health                - 헬스 체크
    POST /api/crawl             - 특정 티커 크롤링 (동기)
    POST /api/debate            - 멀티 에이전트 토론 (동기)
    POST /api/run               - 크롤링 + 토론 통합 실행 (job_id 반환)
    GET  /api/jobs/{job_id}     - 작업 상태 및 결과 조회
    GET  /api/jobs              - 전체 작업 목록 조회
"""

import os
import uuid
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional, Literal
import traceback

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from src.Agent.kanana_pipeline import get_kanana_pipeline
from src.Crawling.crawling_main import main as run_crawling
from src.Agent.graph import agent_debate_graph

PORT_NUM = int(os.getenv("PORT_NUM", "8001"))

# ============================================================================
# 작업(Job) 저장소 (인메모리)
# ============================================================================
jobs: dict[str, dict] = {}
jobs_lock = Lock()

# ============================================================================
# 스레드 풀 (에이전트/크롤러는 동기 코드이므로 별도 스레드에서 실행)
# ============================================================================
executor = ThreadPoolExecutor(max_workers=2)

# ============================================================================
# FastAPI 앱 초기화
# ============================================================================
app = FastAPI(
    title="Stock Agent API",
    description="Kanana 기반 주식 멀티 에이전트 토론 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 요청 / 응답 스키마
# ============================================================================
class TickerRequest(BaseModel):
    ticker: str = Field(..., description=f"기업 티커 (지원: {', '.join(Config.TICKER_MAP.keys())})")


class RunRequest(BaseModel):
    ticker: str = Field(..., description=f"기업 티커 (지원: {', '.join(Config.TICKER_MAP.keys())})")
    mode: Literal["sync", "background"] = Field(
        default="background",
        description="sync: 요청 완료까지 대기 / background: 백그라운드 실행 후 job_id 반환",
    )


class CrawlResponse(BaseModel):
    status: str
    ticker: str


class DebateResponse(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    final_consensus: str


class JobStatusResponse(BaseModel):
    job_id: str
    ticker: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[dict] = None


class RunResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    message: str


# ============================================================================
# 내부 유틸
# ============================================================================
def _normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker는 비어 있을 수 없습니다.")
    return ticker


def _validate_ticker(ticker: str) -> str:
    ticker = _normalize_ticker(ticker)
    if ticker not in Config.TICKER_MAP:
        supported = ", ".join(Config.TICKER_MAP.keys())
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 티커입니다: '{ticker}'. 지원 티커: {supported}",
        )
    return ticker


def _build_initial_state(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "context": "",
        "optimist_initial": "",
        "pessimist_initial": "",
        "debate_history": [],
        "turn_count": 0,
        "max_turns": 6,
        "current_agent": "start",
        "final_consensus": None,
    }


def _run_debate_sync(ticker: str) -> dict:
    """그래프를 동기 실행하고 결과를 반환합니다."""
    graph = agent_debate_graph()
    return graph.invoke(_build_initial_state(ticker))


# ============================================================================
# 백그라운드 작업 함수 (스레드에서 실행)
# ============================================================================
def _run_all_job(job_id: str, ticker: str):
    """크롤링 → 토론을 순차 실행하고 jobs 딕셔너리를 업데이트합니다."""
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        run_crawling(ticker)
        result = _run_debate_sync(ticker)

        final_consensus = result.get("final_consensus") or "합의안 도출에 실패했습니다."

        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "finished_at": datetime.utcnow().isoformat(),
                "result": {
                    "ticker": ticker,
                    "company_name": Config.TICKER_MAP.get(ticker),
                    "final_consensus": final_consensus,
                },
            })

    except Exception as e:
        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "finished_at": datetime.utcnow().isoformat(),
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            })


# ============================================================================
# 라우터
# ============================================================================
@app.get("/", tags=["일반"])
async def root():
    """API 정보를 반환합니다."""
    return {
        "name": "Stock Agent API",
        "version": "1.0.0",
        "description": "Kanana 기반 주식 멀티 에이전트 토론 API",
        "supported_tickers": Config.TICKER_MAP,
        "endpoints": {
            "POST /api/crawl": "특정 티커 뉴스·SEC 데이터 크롤링",
            "POST /api/debate": "멀티 에이전트 토론 실행 (동기)",
            "POST /api/run": "크롤링 + 토론 통합 실행",
            "GET /api/jobs/{job_id}": "작업 상태 및 결과 조회",
            "GET /api/jobs": "전체 작업 목록 조회",
            "GET /health": "헬스 체크",
        },
    }


@app.get("/health", tags=["일반"])
async def health_check():
    """서버 상태를 확인합니다."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/crawl", response_model=CrawlResponse, tags=["파이프라인"])
async def crawl_only(req: TickerRequest):
    """
    지정한 티커의 뉴스 및 SEC 공시 데이터를 크롤링합니다.

    - **ticker**: 기업 티커 (예: NVDA, AAPL)

    크롤링은 동기 방식으로 완료 후 응답을 반환합니다.
    """
    ticker = _validate_ticker(req.ticker)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, run_crawling, ticker)

    return CrawlResponse(status="completed", ticker=ticker)


@app.post("/api/debate", response_model=DebateResponse, tags=["파이프라인"])
async def debate_only(req: TickerRequest):
    """
    지정한 티커에 대한 멀티 에이전트 토론을 실행합니다.

    - **ticker**: 기업 티커 (예: NVDA, AAPL)

    토론은 동기 방식으로 완료 후 최종 합의안을 반환합니다.
    크롤링이 먼저 완료된 상태여야 정확한 분석이 가능합니다.
    """
    ticker = _validate_ticker(req.ticker)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, _run_debate_sync, ticker)

    final_consensus = result.get("final_consensus") or "합의안 도출에 실패했습니다."

    return DebateResponse(
        ticker=ticker,
        company_name=Config.TICKER_MAP.get(ticker),
        final_consensus=final_consensus,
    )


@app.post("/api/run", response_model=RunResponse, tags=["파이프라인"])
async def run_all(req: RunRequest):
    """
    크롤링과 멀티 에이전트 토론을 순차 실행합니다.

    - **ticker**: 기업 티커 (예: NVDA, AAPL)
    - **mode**: `sync` (완료까지 대기) / `background` (즉시 job_id 반환)

    `background` 모드: 응답으로 `job_id`를 반환하며, `GET /api/jobs/{job_id}`로 결과를 폴링하세요.
    """
    ticker = _validate_ticker(req.ticker)

    if req.mode == "sync":
        loop = asyncio.get_event_loop()
        run_crawling_result = await loop.run_in_executor(executor, run_crawling, ticker)
        result = await loop.run_in_executor(executor, _run_debate_sync, ticker)
        final_consensus = result.get("final_consensus") or "합의안 도출에 실패했습니다."
        return RunResponse(
            job_id="sync",
            ticker=ticker,
            status="completed",
            message=final_consensus,
        )

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_all_job, job_id, ticker)

    return RunResponse(
        job_id=job_id,
        ticker=ticker,
        status="queued",
        message=f"작업이 등록되었습니다. GET /api/jobs/{job_id} 로 결과를 확인하세요.",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse, tags=["작업"])
async def get_job(job_id: str):
    """
    작업 상태와 결과를 조회합니다.

    - **status**: `queued` → `running` → `completed` / `failed`
    - **result**: 완료 시 최종 합의안 포함
    - **error**: 실패 시 오류 내용 포함
    """
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}'를 찾을 수 없습니다.")
    return job


@app.get("/api/jobs", tags=["작업"])
async def list_jobs():
    """
    전체 작업 목록과 요약 상태를 반환합니다.
    """
    with jobs_lock:
        snapshot = list(jobs.values())

    return {
        "total": len(snapshot),
        "jobs": [
            {
                "job_id": j["job_id"],
                "ticker": j["ticker"],
                "status": j["status"],
                "created_at": j["created_at"],
                "finished_at": j.get("finished_at"),
            }
            for j in snapshot
        ],
    }


# ============================================================================
# 서버 시작 시 모델 사전 로드
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 Kanana 모델을 사전 로드합니다."""
    loop = asyncio.get_event_loop()
    print("🔄 Kanana 모델 사전 로드 중... (처음 실행 시 몇 분 소요될 수 있습니다)")
    await loop.run_in_executor(executor, get_kanana_pipeline)
    print("✅ Kanana 모델 로드 완료 — API 서버 준비!")


# ============================================================================
# 직접 실행 시
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT_NUM, reload=False)
