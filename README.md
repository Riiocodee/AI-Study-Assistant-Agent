# AI Study Assistant Agent

An **Agentic RAG** application that answers questions from uploaded study PDFs.

## What makes it agentic?

This is not only a basic "retrieve and answer" RAG pipeline. The agent can:

1. Retrieve relevant chunks from the uploaded PDF.
2. Grade whether the retrieved context is relevant to the question.
3. Rewrite/broaden the search query if the first retrieval is not good enough.
4. Retrieve again.
5. Generate the final answer using the retrieved context.

### Flow

```text
User Question
      ↓
Vector Retrieval
      ↓
Relevance Grader
   ↙       ↘
NO          YES
 ↓           ↓
Rewrite      Answer
Query
 ↓
Retrieve Again
```

## Tech Stack

- Python
- LangGraph
- LangChain
- ChromaDB
- OpenAI embeddings + chat model
- Streamlit
- PyPDF

## Run locally

```bash
git clone <your-repository-url>
cd AI-Study-Assistant-Agent

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run app.py
```

## Example

Upload a lecture PDF and ask:

> What is the difference between supervised and unsupervised learning?

The agent retrieves the relevant section, checks the retrieved context, and answers from the notes.

## Project Structure

```text
AI-Study-Assistant-Agent/
├── app.py
├── agent.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
```

## Limitations

The current version uses an OpenAI-compatible API for the LLM and embeddings. It is intended as a small demonstration of an agentic RAG workflow rather than a production deployment.
