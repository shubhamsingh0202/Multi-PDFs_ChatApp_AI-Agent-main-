---
title: Multi-PDF Chat AI Agent
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: chatapp.py
pinned: false
---

# Multi-PDF Chat App

A small Streamlit app for chatting with one or more PDF files. Upload your documents, ask questions in plain English, and get answers backed by the text in those files.

It uses BM25 for retrieval (keyword search, runs locally, no embeddings) and Groq's Llama 3.3 70B model to generate responses.

## What it does

1. Reads uploaded PDFs and pulls out the text.
2. Splits the text into chunks so it fits in the model's context window.
3. Builds a BM25 index over those chunks for fast keyword-based search.
4. When you ask a question, it finds the most relevant chunks and sends them to the LLM along with your question and recent chat history.
5. The model answers using only that context. If the answer isn't in the documents, it says so.

## Features

- Upload and query multiple PDFs at once
- Groq API key via sidebar or a `.env` file
- BM25 indexing — no embedding API calls, no GPU needed
- Keeps the last few turns of conversation for follow-up questions
- Basic validation and error messages in the UI

## Requirements

- Python 3.8+
- A [Groq API key](https://console.groq.com/)

Main dependencies: Streamlit, Groq SDK, LangChain, PyPDF2, rank_bm25, python-dotenv. See `requirements.txt` for the full list.

## Setup

Clone the repo:

```bash
git clone https://github.com/shubhamsingh0202/Multi-PDFs_ChatApp_AI-Agent-main-.git
cd Multi-PDFs_ChatApp_AI-Agent-main-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your API key (optional — you can also enter it in the app sidebar):

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run chatapp.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. Enter your Groq API key in the sidebar if you didn't set up a `.env` file.
2. Upload one or more PDF files.
3. Click **Process Documents** and wait for indexing to finish.
4. When you see the "Ready to Chat" status, type a question and hit Enter.
5. Use **Clear Chat** in the sidebar to reset the conversation.

## How it's built

| Piece | Role |
|-------|------|
| Streamlit | Web UI |
| PyPDF2 | PDF text extraction |
| LangChain | Text splitting and BM25 retriever |
| Groq (`llama-3.3-70b-versatile`) | Answer generation |

## License

MIT — see [LICENSE](LICENSE) for details.
