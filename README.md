# Valura AI Microservice

Valura is an AI-powered backend microservice built for a wealth management platform. The idea behind it is to act as a smart co-investor that can understand what a user is asking, check if it is safe, figure out the intent, and then route it to the right specialized agent. Right now only the portfolio health analysis part is fully built out, but the architecture is designed so that adding more agents later is straightforward.

This was built as part of an internship project. The goal was to go beyond just making something work and actually think about latency, cost, and safety from the beginning.

---

## What it does

When a user sends a query, the service runs it through a pipeline in this order:

1. **Safety Guard** - Checks the query for harmful intent before anything else. This runs purely with regex patterns (no LLM call needed), so it stays under 10ms. Categories it blocks include insider trading, market manipulation, pump-and-dump schemes, money laundering, and reckless advice like going all-in on a single stock.

2. **Intent Classification** - If the query passes safety, an LLM call (OpenAI) figures out what the user actually wants. It extracts structured data from the query like ticker symbols, dollar amounts, time periods, and sectors. It also uses recent conversation history so it can handle follow-up queries that reference previous turns.

3. **Agent Routing** - Based on the classification result, the router sends the query to the right agent. If the agent is not yet implemented, it returns a stub response instead of crashing.

4. **Portfolio Health Agent** - This is the only fully implemented agent. It fetches live market data using yfinance, calculates concentration risk, computes total return vs purchase price, compares performance against an appropriate benchmark (SPY, EFA, or VT depending on portfolio composition), and generates plain-language observations for the user.

Everything streams back to the client using Server-Sent Events (SSE), so the frontend can show progress in real time.

---

## Project structure

```
Valura/
├── src/
│   ├── main.py              # FastAPI app, SSE pipeline, endpoints
│   ├── models.py            # All Pydantic data models
│   ├── config.py            # Config, env vars, agent taxonomy
│   ├── core/
│   │   ├── classifier.py    # Intent classification using OpenAI
│   │   ├── router.py        # Routes classified queries to agents
│   │   ├── safety_guard.py  # Regex-based safety filtering
│   │   └── session.py       # Conversation history via SQLite
│   ├── agents/
│   │   ├── base.py          # Abstract base class for all agents
│   │   └── portfolio_health.py  # Portfolio analysis agent
│   └── utils/
│       ├── llm_client.py    # OpenAI wrapper with retry logic
│       ├── metrics.py       # Latency and cost tracking
│       └── streaming.py     # SSE helpers
└── tests/
    ├── conftest.py              # Shared fixtures and mocks
    ├── test_classifier.py       # Classifier unit tests
    ├── test_matching.py         # Entity matching tests
    ├── test_portfolio_health.py # Portfolio agent tests
    └── test_safety_guard.py     # Safety guard tests
```

---

## Setup

### Requirements

- Python 3.10 or higher
- An OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are FastAPI, Uvicorn, OpenAI, yfinance, Pydantic, and python-dotenv.

### Environment variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini        # optional, defaults to gpt-4o-mini in dev
ENV=development                  # or production
DEBUG=true
```

In production mode, `OPENAI_API_KEY` is required and the app will raise an error if it is missing. In development, you can test most things without a real key by using the mocked tests.

### Run the server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API

### POST /query

Main endpoint. Accepts a user query along with their context and streams back events.

Request body:

```json
{
  "query": "How is my portfolio doing?",
  "user_context": {
    "user_id": "user_001",
    "session_id": "session_abc",
    "profile": {
      "user_id": "user_001",
      "risk_profile": "moderate",
      "kyc_status": "approved",
      "country": "US",
      "preferred_currency": "USD",
      "investment_goals": []
    },
    "portfolio": [
      {
        "ticker": "AAPL",
        "quantity": 10,
        "purchase_price": 150.0
      },
      {
        "ticker": "MSFT",
        "quantity": 5,
        "purchase_price": 300.0
      }
    ]
  }
}
```

The `risk_profile` field accepts: `conservative`, `moderate`, `aggressive`, or `very_aggressive`.

The response is a stream of SSE events. Each event has a name and a JSON data payload. The sequence looks like this:

```
event: safety_check
data: {"status": "running"}

event: safety_passed
data: {"status": "safe"}

event: classification_complete
data: {"intent": "...", "agent": "portfolio_health", "confidence": 0.95, ...}

event: agent_complete
data: {"time_taken": 2.3}

event: result
data: {"agent": "portfolio_health", "response": {...}, "classification": {...}}

event: metrics
data: {"total_time": 3.1, "classification_time": 0.8, "agent_time": 2.3}

event: done
data: {}
```

If safety blocks the query, a `safety_blocked` event fires and the stream ends immediately. If anything times out, an `error` event is sent instead.

### GET /health

Returns the current status of the service, the model being used, and the environment.

```json
{
  "status": "healthy",
  "model": "gpt-4o-mini",
  "environment": "development",
  "version": "0.1.0"
}
```

---

## How the portfolio health agent works

When the portfolio health agent gets called, it does the following:

- Fetches current price, sector, and one year of price history for all holdings from yfinance. This happens concurrently so it does not have to wait for each ticker one by one.
- Calculates concentration risk by looking at what percentage of total portfolio value is in the top position and the top three positions. Flags range from `low` to `extreme`.
- Calculates total return based on purchase price vs current price. Annualized return is noted as a TODO since that requires purchase dates to do properly.
- Detects the right benchmark automatically. If most tickers look like US stocks (no exchange suffix like `.L` or `.TO`), it uses SPY. If mostly international, it uses EFA. Mixed portfolios get compared to VT.
- Generates a short list of plain-language observations capped at four, covering concentration issues, performance summary, benchmark comparison, and sector concentration.

If the portfolio is empty, it skips all of that and returns a "build mode" message based on the user's risk profile to nudge them toward getting started.

---

## Safety guard details

The safety guard is intentionally kept separate from the LLM to avoid adding latency. It uses compiled regex patterns organized by five categories:

- `insider_trading` - patterns like "non-public information", "MNPI", "friend works at [company]"
- `market_manipulation` - patterns like "pump and dump", "artificially inflate"
- `guaranteed_returns` - patterns like "risk-free profit", "can't lose", "100% guaranteed"
- `money_laundering` - patterns like "launder funds", "evade taxes", "unreported income"
- `reckless_advice` - patterns like "max leverage", "mortgage house to invest", "crypto leverage 10x"

Before checking those patterns, it checks if the query looks educational. Questions like "what is insider trading?" or "explain pump and dump schemes" should be allowed through because the user is trying to learn, not act. This detection looks for question words, "explain", "define", "how does X work", and similar phrasing.

---

## Session management

The service stores conversation history in a SQLite database (path configured via `DB_PATH` in `.env`). Each session stores the sequence of query-agent-response turns. When the classifier runs, it receives the last three turns as context so it can handle follow-up questions like "how does it compare to Google?" after previously discussing Microsoft.

Sessions are capped at 10 turns by default (`SESSION_MAX_TURNS`). Older turns are automatically cleaned up once the limit is exceeded.

---

## Performance targets

These are the targets the system is designed to meet:

| Metric | Target |
|---|---|
| Safety guard latency | under 10ms |
| Classification p95 latency | under 2 seconds |
| Total pipeline p95 latency | under 6 seconds |
| Mean cost per query | under $0.05 |

The `MetricsTracker` class in `utils/metrics.py` tracks all of this across queries and can print a summary showing p50, p95, and p99 latencies along with per-agent breakdowns.

The dev model is `gpt-4o-mini` which is significantly cheaper. Production uses `gpt-4.1`. You can override the model via the `OPENAI_MODEL` env var.

---

## Agents available

| Agent | Status | Description |
|---|---|---|
| `portfolio_health` | Implemented | Analyzes portfolio composition, risk, and performance |
| `market_research` | Stub | Market data and company research |
| `investment_strategy` | Stub | Building investment plans |
| `financial_calculator` | Stub | Compound interest and financial calculations |
| `risk_assessment` | Stub | Evaluating investment risks |
| `recommendation` | Stub | Specific investment action suggestions |
| `support` | Stub | General questions and educational queries |

Stub agents return a structured response with a message explaining the feature is not yet available, rather than returning an error. The classification still runs correctly for these, so the pipeline does not break.

---

## Running tests

```bash
pytest tests/ -v
```

The tests are designed to run without a real OpenAI API key or live market data. The `conftest.py` file contains fixtures that mock the OpenAI client and yfinance so tests do not make any real network calls.

There are also `slow` and `integration` markers for tests that require real API access. These are skipped by default. To run only unit tests:

```bash
pytest tests/ -v -m unit
```

---

## Adding a new agent

To add a new agent, the steps are:

1. Create a new file in `src/agents/` that inherits from `BaseAgent` and implements the `execute()` method.
2. Register the agent in `src/core/router.py` by adding it to the `_agents` dictionary.
3. Update `IMPLEMENTED_AGENTS` in `config.py`.

The taxonomy and description for each agent already lives in `config.py` under `AGENT_TAXONOMY`. The classifier already knows about all agents and will route to them correctly once they are registered in the router.

---

## Notes

This is still an early build. A few things worth mentioning:

- Annualized return calculation is incomplete. It currently just shows total return since purchase. Proper time-weighted return calculation is noted as a TODO in the portfolio health agent.
- The session database uses SQLite which works fine for development and small loads. A production deployment would want to evaluate whether a more scalable store is needed.
- The OpenAI API key is loaded from environment variables. Never commit a real key to version control.
- Cost tracking is logged to the console in debug mode. The `MetricsTracker` can save this to a JSON file for offline analysis.
