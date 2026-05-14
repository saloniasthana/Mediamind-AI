# Mediamind-AI

AI-powered document and multimedia Q&A web application for the SDE-1 programming assignment.

## Features

- Upload PDF, text, audio, and video files.
- Extract readable content from PDFs and transcripts from media.
- Ask questions against uploaded content.
- Generate summaries for every uploaded file.
- Find topic-specific timestamps in audio and video.
- Play media from the timestamp used by an answer or topic result.
- FastAPI backend, React frontend, SQLite persistence, Docker Compose, GitHub Actions CI, and 95% coverage gate.

## Tech Stack

- Backend: Python, FastAPI, MongoDB, PyMongo.
- Frontend: React, Vite, lucide-react.
- AI layer: local deterministic retrieval/summarization by default, structured so OpenAI, Whisper, LangChain, LlamaIndex, FAISS, Pinecone, or Deepgram can be plugged in.
- Testing: pytest, pytest-cov with `--cov-fail-under=95`.

## Run Locally

```bash
docker compose up --build
```

Frontend: http://localhost:5173  
Backend docs: http://localhost:8000/docs

## Run Without Docker

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

For non-Docker backend runs, make sure MongoDB is available locally at `mongodb://localhost:27017`, or set:

```bash
MONGODB_URL=mongodb://your-host:27017
MONGODB_DATABASE=mediamind
```

For high-quality PDF Q&A, add a Groq key before starting the backend:

```bash
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

OpenAI is also supported as a fallback with `OPENAI_API_KEY` and `OPENAI_MODEL`. Without an LLM key, the app uses a local fallback search answer, which is useful for demos but less accurate than a hosted model.

## API

- `GET /health` - service health.
- `POST /api/upload` - multipart upload field `file`.
- `GET /api/documents` - list uploaded documents and summaries.
- `POST /api/chat` - body `{ "question": "...", "document_id": 1 }`.
- `POST /api/timestamps` - body `{ "topic": "...", "document_id": 1 }`.
- `GET /api/media/{document_id}` - stream uploaded media for playback.

## Testing

```bash
cd backend
pytest
```

The repository enforces at least 95% backend test coverage through `backend/pytest.ini` and CI.

## Real AI Integration Notes

The current implementation is demo-ready and test-friendly without paid API keys. For production:

- Replace `app/services/qa.py` with OpenAI, LangChain, LlamaIndex, FAISS, or Pinecone retrieval generation.
- Replace `transcribe_media` in `app/services/extraction.py` with Whisper API, OpenAI ASR, or Deepgram.
- Move from the current MongoDB document store to Elasticsearch, Pinecone, or a dedicated vector database depending on scale.
- Add JWT/OAuth auth, Redis rate limiting, and response streaming for bonus points.

## Walkthrough Video Checklist

1. Start the stack with Docker Compose.
2. Upload one PDF and one audio or video file.
3. Show the generated summary.
4. Ask a question in chat.
5. Click a cited play button.
6. Search a media topic and play the timestamp result.
7. Show backend tests passing with coverage.
