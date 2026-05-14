from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.database import MongoStore, get_database, init_db
from app.models import Chunk, Document
from app.schemas import ChatRequest, ChatResponse, DocumentOut, TimestampResult, TopicRequest
from app.services.extraction import extract_upload, save_upload
from app.services.qa import answer_question, search_chunks

app = FastAPI(title="MediaMind AI", version="1.0.0")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload", response_model=DocumentOut)
async def upload(file: UploadFile = File(...), db: MongoStore = Depends(get_database)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="File content type is required")
    path = await save_upload(file)
    text, summary, chunks = extract_upload(path, file.content_type)
    document = Document(
        filename=file.filename,
        content_type=file.content_type,
        file_path=str(path),
        extracted_text=text,
        summary=summary,
    )
    document = db.add_document(document)
    for item in chunks:
        db.add_chunk(Chunk(document_id=document.id, **item))
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        summary=document.summary,
    )


@app.get("/api/documents", response_model=list[DocumentOut])
def documents(db: MongoStore = Depends(get_database)):
    rows = db.list_documents()
    return [
        DocumentOut(id=row.id, filename=row.filename, content_type=row.content_type, summary=row.summary)
        for row in rows
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: MongoStore = Depends(get_database)):
    answer, citations = answer_question(db, payload.question, payload.document_id)
    return ChatResponse(answer=answer, citations=citations)


@app.post("/api/timestamps", response_model=list[TimestampResult])
def timestamps(payload: TopicRequest, db: MongoStore = Depends(get_database)):
    matches = search_chunks(db, payload.topic, payload.document_id, limit=5)
    results = []
    for chunk, document in matches:
        if document.content_type.startswith(("audio/", "video/")):
            results.append(
                TimestampResult(
                    document_id=document.id,
                    filename=document.filename,
                    topic=payload.topic,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    snippet=chunk.text,
                )
            )
    return results


@app.get("/api/media/{document_id}")
def media(document_id: int, db: MongoStore = Depends(get_database)):
    document = db.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(document.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(path, media_type=document.content_type, filename=document.filename)
