# SD40-2 Locomotive Manual RAG Assistant

A retrieval-augmented generation (RAG) pipeline that lets you ask natural language questions against the SD40-2 Locomotive Service Manual (416 pages) and get accurate, sourced answers.

Built with LangChain, FAISS, and Ollama (llama3.2 + nomic-embed-text) — runs fully locally, no API key needed.

## Demo

![App Screenshot](screenshot.png)

## Evaluation Results

| Metric | Result |
|--------|--------|
| Questions answered | 50 / 50 (100%) |
| Average response time | 3.25s |
| Total pages indexed | 416 |
| Total chunks | 2,123 |

## How it works

1. `ingest.py` loads the PDF, splits it into 500-character chunks, embeds them with `nomic-embed-text`, and saves the FAISS index to disk
2. `app.py` loads the index, takes a user question, retrieves the 4 most relevant chunks, and passes them to `llama3.2` to generate an answer
3. `evaluate.py` runs 50 test questions and measures answer rate and response time

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your PDF to the `docs/` folder and update `PDF_PATH` in `ingest.py`

4. Run ingestion:
```bash
python src/ingest.py
```

5. Run the app:
```bash
streamlit run src/app.py
```

## Stack

- LangChain 0.3.25
- FAISS
- Ollama (llama3.2, nomic-embed-text)
- Streamlit
- Python 3.13