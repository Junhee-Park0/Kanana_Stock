# Stock News: Crawling + Multi-Agent Debate

뉴스/SEC 데이터를 수집하고, 이를 기반으로 멀티 에이전트 토론(낙관 vs 비관 vs 중재)을 수행해 투자 관점의 최종 합의안을 생성하는 프로젝트입니다.

## 주요 기능

- `src/Crawling`: Yahoo Finance 뉴스 + SEC 공시 수집/파싱 및 SQLite 저장
- `src/Agent`: LangGraph 기반 멀티 에이전트 토론 워크플로우
- `main.py`: 크롤링 -> 토론을 순차 실행하는 통합 CLI 진입점
- `api.py`: FastAPI 백엔드 (`crawl`, `debate`, `run-all`, `jobs`)

## 현재 프로젝트 구조

```text
Stock_News/
├── main.py
├── api.py
├── requirements.txt
├── config.py
└── src/
    ├── Crawling/
    │   ├── crawling_main.py
    │   ├── news_crawling.py
    │   ├── news_db.py
    │   ├── sec_crawling.py
    │   ├── sec_parsing.py
    │   ├── sec_db.py
    │   └── get_context.py
    └── Agent/
        ├── agent_main.py
        ├── graph.py
        ├── nodes.py
        ├── functions.py
        ├── tools.py
        ├── kanana_pipeline.py
        ├── prompts.yaml
        └── states.py
```

## 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

`.env`에 실행에 필요한 환경변수를 설정합니다.

- 로컬 Kanana 모델 경로/옵션은 `config.py`를 기준으로 사용합니다.
- SEC 크롤링을 위한 User-Agent/이메일 설정이 필요할 수 있습니다.

## CLI 실행 방법

### 1) 크롤링 + 토론 통합 실행

```bash
python -m main --ticker NVDA
```

### 2) 크롤링만 실행

```bash
python -m src.Crawling.crawling_main --ticker NVDA
```

### 3) 에이전트 토론만 실행

```bash
python -m src.Agent.agent_main --ticker NVDA
```

## FastAPI 백엔드

서버 실행:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

주요 엔드포인트:

- `GET /health`: 헬스체크
- `POST /crawl`: 크롤링만 실행
- `POST /debate`: 토론만 실행, `final_consensus` 반환
- `POST /run-all`: 크롤링+토론 통합 실행 (`sync`/`background`)
- `GET /jobs/{job_id}`: 백그라운드 작업 상태 조회

요청 예시:

```json
{
  "ticker": "NVDA",
  "mode": "background"
}
```

## 주의사항

- 에이전트는 로컬 모델 추론 환경(CPU/GPU, 메모리)에 따라 응답 시간이 크게 달라질 수 있습니다.
- 크롤링은 대상 사이트 구조 변경 또는 드라이버 이슈에 영향을 받을 수 있습니다.
- `python src/Agent/agent_main.py ...` 형태보다 `python -m src.Agent.agent_main ...` 방식 실행을 권장합니다.
