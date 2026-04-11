# MAS-Briefing: Multi-Agent Investment Briefing System

**MAS-Briefing**은 LLM 기반의 멀티 에이전트 시스템을 활용하여 특정 주식 종목에 대한 심층적인 투자 분석과 브리핑을 제공하는 프로젝트입니다. 뉴스 기사와 SEC 공시 데이터를 실시간으로 수집하고, 이를 바탕으로 다각적인 분석(낙관론 vs 비관론 토론)을 수행하여 최종 합의된 투자 의견을 도출합니다.

## 📌 Key Features

### 1. Advanced Data Pipeline
데이터 수집은 크게 두 가지 경로를 통해 이루어지며, 분석에 필요한 풍부한 Context를 확보합니다.

*   **Yahoo Finance Crawling**:
    *   `Crawling` 모듈을 통해 Yahoo Finance의 종목별 최신 뉴스를 직접 크롤링합니다.
    *   수집된 뉴스(제목, 본문, 저자 등)는 로컬 **SQLite DB**(`News_DB`)에 체계적으로 저장되어 관리됩니다.
*   **AWS & SEC Data Fetching**:
    *   `Fetch_Data` 모듈을 활용하여 AWS DynamoDB에 적재된 뉴스 데이터와 SEC EDGAR 시스템의 기업 공시(Filings) 데이터를 가져옵니다.
    *   뉴스뿐만 아니라 공시 데이터까지 통합하여 에이전트가 팩트에 기반한 정교한 분석을 할 수 있도록 지원합니다.

### 2. Multi-Agent Debate (Core)
수집된 데이터를 바탕으로 서로 다른 페르소나를 가진 에이전트들이 토론을 진행합니다.

*   **낙관론자(Optimist)**: 시장의 기회 요인과 긍정적 지표(호재성 뉴스, 성장 가능성)를 중심으로 분석
*   **비관론자(Pessimist)**: 잠재적 리스크와 부정적 지표(규제 이슈, 실적 우려)를 중심으로 분석
*   **중재자(Neutral)**: 양측의 치열한 토론 내용을 종합하여 편향되지 않은 균형 잡힌 최종 합의(Consensus) 도출
*   **LangGraph Workflow**: `Initial Opinion` → `Debate Loop` (상호 반박) → `Summary` → `Save`의 순차적 흐름을 통해 논리적 완성도를 높입니다.

### 3. Single Agent Briefing
*   복잡한 토론 과정 없이, 수집된 데이터를 요약하여 투자자를 위한 간결한 일일 브리핑 리포트를 생성하는 기능도 제공합니다.

## 📂 Project Structure

```
MAS_Project/
├── crawling_main.py       # [Crawling] Yahoo Finance 뉴스 크롤링 및 DB 저장 실행
├── single_agent_main.py   # [Single Agent] 브리핑 생성 실행
├── multi_agent_main.py    # [Multi Agent] 데이터 통합 및 토론 실행
├── src/
│   ├── Crawling/          # Yahoo Finance 크롤링 및 SQLite DB 핸들링 로직
│   ├── Fetch_Data/        # AWS DynamoDB 뉴스 및 SEC 공시 데이터 Fetching 로직
│   ├── Single_Agent/      # 싱글 에이전트 로직
│   └── Multi_Agent/       # 멀티 에이전트 토론 로직 (LangGraph)
│       ├── graph.py       # 에이전트 워크플로우 정의
│       ├── nodes.py       # 낙관/비관/중재자 노드 정의
│       └── prompts.yaml   # 각 에이전트의 페르소나 및 프롬프트
└── data/
    ├── News_DB/           # 크롤링된 뉴스가 저장되는 SQLite DB
    ├── SEC/               # 수집된 SEC 공시 데이터
    ├── Briefings/         # 생성된 브리핑 결과물
    └── Debate/            # 멀티 에이전트 토론 로그 및 최종 합의문
```

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.8+
*   OpenAI API Key (GPT-4o 활용)

### 2. Installation

```bash
# 프로젝트 클론
git clone [repository_url]

# 패키지 설치
pip install -r requirements.txt
```

### 3. Configuration
프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 설정합니다.

```env
OPENAI_API_KEY=sk-proj-xxxx...
```

## 💻 Usage

### 1. Data Collection (Crawling)
Yahoo Finance에서 최신 뉴스를 크롤링하여 로컬 DB에 저장합니다.
```bash
# 사용법: python crawling_main.py [TICKER] [COUNT]
python crawling_main.py "NVDA" 5
```

### 2. Multi-Agent Debate (토론 및 합의)
저장된 뉴스 데이터와 SEC 데이터를 통합(`Fetch_Data`)하여 토론을 진행합니다.
```bash
# 사용법: python multi_agent_main.py --ticker [TICKER] --keywords [KEYWORDS...]
python multi_agent_main.py --ticker NVDA --keywords "AI" "Data Center" "H100"
```
*   **Process**: 데이터 수집/통합 -> 낙관/비관 초기 의견 -> 상호 반박 -> 최종 합의
*   **Output**: `data/Debate/` 폴더에 결과 저장

### 3. Single Agent Briefing
(Optional) 단일 에이전트가 작성한 브리핑을 생성합니다.
```bash
python single_agent_main.py --ticker NVDA
```

## 🛠 Tech Stack
*   **Language**: Python
*   **Framework**: LangChain, LangGraph
*   **LLM**: OpenAI GPT-4o
*   **Data Source**: Yahoo Finance (Crawling), AWS DynamoDB, SEC EDGAR
*   **Database**: SQLite
