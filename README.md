# Multi-Tool AI Agent (LangGraph + OpenAI + FAISS)

A general-purpose intelligent agent powered by LangGraph, OpenAI’s GPT-4o, and a suite of custom tools.
The agent autonomously decides when to call tools such as web search, Wikipedia search, scientific paper lookup, FAISS retrieval, and a calculator - combining them with LLM reasoning to produce accurate, grounded answers.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/langgraph-✓-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

<p align="center">
  <img src="data/gaia.png" alt="gaia" width="500"/>
</p>

---

## 🔍 Features

- **Autonomous Tool Calling**: The agent decides when and which tools to use (RAG, web search, Wikipedia, ArXiv, calculator) using a LangGraph ReAct-style workflow.

- **LangGraph State Machine Orchestration**: A transparent, modular graph design manages reasoning loops, tool calls, and termination conditions - easy to extend or customize.

- **Retrieval-Augmented Generation (RAG)**: Built-in FAISS index over local Q&A examples for grounded responses without external APIs.

- **Optional Real-Time Web Search**: Integrates Tavily for fresh, up-to-date information; automatically degrades gracefully if no API key is provided.

- **Rich Tooling Ecosystem**: Includes Wikipedia and ArXiv search utilities, plus safe arithmetic evaluation - all wrapped as LangChain tools.

- **Clean Gradio Chat Interface**: Modern chat UI with tool-usage visibility, making the agent’s reasoning process more transparent.

- **Fully Open-Source & Extensible**: Designed for learning, experimentation, and showcasing AI agent best practices.

---

## 📁 Project Structure

```bash
.
├── agent.py          # LangGraph pipeline + tool routing logic
├── app.py            # Gradio UI (chat-style) with tool-call display
├── tools.py          # Custom tools (RAG, web search, wiki, arxiv, calculator)
├── data/
│   └── metadata.jsonl  # Q&A examples used for local FAISS retrieval
├── constants.py      # Optional system prompt
├── .env              # API keys (not committed)
├── requirements.txt  # Dependencies
````

---

## 🔧 Setup (Local)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gaia-ai-agent.git
cd gaia-ai-agent
```

### 2. Create and activate environment

```bash
conda create -n agent python=3.11
conda activate agent
uv pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=optional_web_search_key
```

Web search is optional: if no Tavily key is provided, the tool gracefully disables itself.

### 4. Run CLI

```bash
python agent.py "Who discovered penicillin?"
```

or:

```bash
python agent.py "Summarize the main ideas of diffusion models."
```

### 5. Launch the Gradio Chat UI

```bash
python app.py
```

---

## 🧠 Tools

| Tool                 | Description                                  |
| -------------------- | -------------------------------------------- |
| **similar_question** | Retrieves similar Q&A examples via FAISS RAG |
| **web_search**       | Up-to-date search using Tavily (optional)    |
| **wiki_search**      | Fetches relevant Wikipedia snippets          |
| **arxiv_search**     | Academic paper search (ArXiv)                |
| **calculator**       | Safe arithmetic computation                  |

The agent selects tools automatically using a ReAct-style LangGraph workflow.

---

## 🔒 Environment Variables

| Variable         | Purpose                            |
| ---------------- | ---------------------------------- |
| `OPENAI_API_KEY` | Required – model inference         |
| `TAVILY_API_KEY` | Optional – enables real web search |

---

## 📝 License
This project is licensed under the MIT License.
