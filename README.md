# Morning Report Agent

An AI agent that generates and emails a personalised morning briefing every day.
It aggregates your **Google Calendar schedule**, **recent unread emails**, and **top news** for your chosen topics and location - then uses Claude to synthesise everything into a clean HTML email delivered straight to your inbox.

![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/langgraph-✓-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## How it works

The agent is built with **LangGraph** and follows a fan-out → fan-in workflow:

```
                        START
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   fetch_calendar    fetch_emails     fetch_news     ← run in parallel
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                   generate_report              ← Claude synthesises the data
                          │
                      send_email               ← delivered via Gmail
                          │
                         END
```

The three data-fetching nodes run **in parallel** - LangGraph fans them out from `START` and automatically waits for all three to complete before the report is generated.  Each node writes into a shared `MorningReportState` using append reducers, so parallel writes never overwrite each other.

---

## Features

- **Parallel data fetching** - calendar, email, and news are fetched simultaneously via LangGraph's fan-out pattern, minimising total runtime.
- **Google Calendar integration** - lists today's events with times and locations.
- **Gmail integration** - surfaces unread emails from the last 24 hours (metadata only - no full body downloads).
- **News aggregation** - queries [NewsAPI](https://newsapi.org) per topic and deduplicates results across topics.
- **LLM-powered synthesis** - Claude reads the raw data and writes a concise, well-formatted HTML briefing.
- **Email delivery** - the finished report is sent via the Gmail API from your own account.
- **Graceful degradation** - any fetch node that fails logs a warning and returns an empty list; the report is still generated and sent with whatever data is available.

---

## Project Structure

```
gaia-ai-agent/
├── main.py                   # CLI entry point
├── src/
│   ├── config.py             # UserConfig - loads settings from .env
│   ├── agent/
│   │   ├── state.py          # MorningReportState TypedDict + sub-record types
│   │   ├── nodes.py          # Node functions (fetch_*, generate_report, send_email)
│   │   └── graph.py          # LangGraph StateGraph definition
│   └── tools/
│       ├── google_auth.py    # Shared OAuth2 credential manager
│       ├── calendar.py       # Google Calendar API wrapper
│       ├── email.py          # Gmail read + send wrapper
│       └── news.py           # NewsAPI wrapper
├── requirements.txt
├── pyproject.toml            # Ruff linter config
├── .env.example              # Template for required environment variables
└── LICENCE
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/gaia-ai-agent.git
cd gaia-ai-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys (see the table below).

### 3. Set up Google OAuth

The agent reads your Calendar and Gmail using the official Google APIs with OAuth 2.0.

1. Go to [Google Cloud Console](https://console.cloud.google.com) and create a project.
2. Enable the **Google Calendar API** and **Gmail API**.
3. Create an **OAuth 2.0 Client ID** (Application type: *Desktop app*).
4. Download the JSON file and save it as **`credentials.json`** in the project root.

On the first run a browser window will open for you to grant consent.  The token is cached in `token.json` so subsequent runs are fully automated.

### 4. Get a NewsAPI key *(optional)*

Sign up for a free key at [newsapi.org/register](https://newsapi.org/register) and add it to `.env` as `NEWS_API_KEY`.  The news section is skipped gracefully if no key is provided.

### 5. Run

```bash
python main.py
```

---

## Environment Variables

### LLM provider

Set **one** of the two API keys below (or both - Anthropic is preferred when both are present).

| Variable           | Description                                              |
|--------------------|----------------------------------------------------------|
| `ANTHROPIC_API_KEY`| [Anthropic](https://console.anthropic.com) key - uses Claude |
| `OPENAI_API_KEY`   | [OpenAI](https://platform.openai.com/api-keys) key - uses GPT |
| `LLM_PROVIDER`     | Optional override: `"anthropic"` or `"openai"` (auto-detected otherwise) |
| `LLM_MODEL`        | Optional model override (defaults: `claude-sonnet-4-6` / `gpt-4o`) |

### Other settings

| Variable           | Required | Description                                              |
|--------------------|----------|----------------------------------------------------------|
| `RECIPIENT_EMAIL`  | ✅       | Email address the report is delivered to                 |
| `USER_NAME`        | -        | Your name, used in the greeting (default: `Friend`)      |
| `NEWS_API_KEY`     | -        | [NewsAPI](https://newsapi.org) key; news skipped if absent |
| `NEWS_TOPICS`      | -        | Comma-separated topics, e.g. `technology,AI,finance`     |
| `NEWS_LOCATION`    | -        | ISO country code for headlines, e.g. `us`, `gb` (default: `us`) |

---

## Tech Stack

| Component      | Library / Service                         |
|----------------|-------------------------------------------|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM            | [Claude](https://www.anthropic.com) or [GPT](https://openai.com) - auto-selected from available key |
| Calendar       | Google Calendar API (`google-api-python-client`) |
| Email          | Gmail API (`google-api-python-client`)    |
| News           | [NewsAPI](https://newsapi.org)            |
| Auth           | Google OAuth 2.0 (`google-auth-oauthlib`) |

---

## License

MIT - see [LICENCE](LICENCE).
