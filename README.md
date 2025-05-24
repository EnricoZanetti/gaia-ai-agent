# GAIA AI Agent (LangGraph + OpenAI + FAISS)

An intelligent agent designed to solve GAIA Level‑1 benchmark questions using retrieval-augmented generation (RAG), OpenAI's GPT-4o, and LangGraph orchestration.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/langgraph-✓-blue)
![HuggingFace](https://img.shields.io/badge/huggingface-spaces-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🔍 Features

- Uses OpenAI `gpt-4o` for reasoning
- FAISS-based retrieval from solved examples
- Additional tools: Wikipedia search, web search, calculator, Arxiv query
- LangGraph state machine for managing LLM/tool orchestration
- Gradio app for evaluation and leaderboard submission (HF Space ready)

---

## 📁 Project Structure

```bash
.
├── agent.py          # Agent logic and LangGraph pipeline
├── app.py            # Gradio UI for running evaluation
├── tools.py          # All custom tools used by the agent
├── metadata.jsonl    # Training data for vector store
├── .env              # Local environment variables (not committed)
├── requirements.txt  # Python dependencies
````

---

## 🚀 Try it on Hugging Face Spaces

[![HF Space](https://img.shields.io/badge/HuggingFace-Live%20Demo-orange)](https://huggingface.co/spaces/enricozan/gaia-ai-agent/tree/main)

---

## 🔧 Setup (Local)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gaia-ai-agent.git
cd gaia-ai-agent
```

### 2. Create and activate environment

```bash
conda create -n gaia python=3.11
conda activate gaia
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_key
GAIA_API_BASE=https://agents-course-unit4-scoring.hf.space
SPACE_HOST=your_huggingface_username
AGENT_CODE_URL=https://huggingface.co/spaces/your-username/your-space-name/tree/main
```

### 4. Run CLI

```bash
python agent.py "What city hosted Expo 2015?"
```

Or evaluate and submit:

```bash
python agent.py --submit
```

### 5. Run Gradio app

```bash
python app.py
```

---

## 🧠 Tools

* `similar_questions`: Retrieve similar solved Q\&A from FAISS
* `wiki_search`: Fetch Wikipedia snippets
* `web_search`: Use Tavily search engine
* `arvix_search`: Academic paper search (Arxiv)
* `calculator`: Basic math eval

---

## 📝 License

This project is licensed under the MIT License.
