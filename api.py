from datetime import datetime
from threading import Lock
from typing import Dict, Literal
from uuid import uuid4
import traceback

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.Crawling.crawling_main import main as run_crawling
from src.Agent.graph import agent_debate_graph


app = FastAPI(
    title="Stock News Pipeline API",
    description="크롤링 + 멀티 에이전트 토론 백엔드",
    version="1.0.0",
)


class RunRequest(BaseModel):
    ticker: str = Field(..., description="기업 티커 (예: NVDA)")


class RunAllRequest(BaseModel):
    ticker: str = Field(..., description="기업 티커 (예: NVDA)")
    mode: Literal["sync", "background"] = Field(
        default="background",
        description="sync: 요청-응답 동안 실행 / background: 백그라운드 실행",
    )


class DebateResponse(BaseModel):
    ticker: str
    final_consensus: str


JOBS: Dict[str, Dict] = {}
JOBS_LOCK = Lock()


def _normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker는 비어 있을 수 없습니다.")
    return ticker


def _run_debate_and_get_result(ticker: str) -> Dict:
    initial_state = {
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
    graph = agent_debate_graph()
    return graph.invoke(initial_state)


def _run_all_job(job_id: str, ticker: str):
    started_at = datetime.utcnow().isoformat()
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = started_at

    try:
        run_crawling(ticker)
        result = _run_debate_and_get_result(ticker)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
            JOBS[job_id]["result"] = {
                "ticker": ticker,
                "final_consensus": result.get("final_consensus"),
            }
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
            JOBS[job_id]["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/crawl")
def crawl_only(req: RunRequest):
    ticker = _normalize_ticker(req.ticker)
    run_crawling(ticker)
    return {"status": "completed", "ticker": ticker, "task": "crawl"}


@app.post("/debate", response_model=DebateResponse)
def debate_only(req: RunRequest):
    ticker = _normalize_ticker(req.ticker)
    result = _run_debate_and_get_result(ticker)
    return {
        "ticker": ticker,
        "final_consensus": result.get("final_consensus", ""),
    }


@app.post("/run-all")
def run_all(req: RunAllRequest, background_tasks: BackgroundTasks):
    ticker = _normalize_ticker(req.ticker)

    if req.mode == "sync":
        run_crawling(ticker)
        result = _run_debate_and_get_result(ticker)
        return {
            "status": "completed",
            "ticker": ticker,
            "result": {"final_consensus": result.get("final_consensus")},
        }

    job_id = str(uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "ticker": ticker,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    background_tasks.add_task(_run_all_job, job_id, ticker)
    return {"status": "queued", "job_id": job_id, "ticker": ticker}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id를 찾을 수 없습니다.")
    return job


@app.get("/")
def root():
    return {
        "message": "Stock News FastAPI backend",
        "endpoints": ["/health", "/crawl", "/debate", "/run-all", "/jobs/{job_id}"],
    }
