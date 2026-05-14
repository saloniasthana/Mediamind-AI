from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader

from app.config import get_settings
from app.services.text import split_chunks, summarize_text, tokenize


async def save_upload(file: UploadFile) -> Path:
    settings = get_settings()
    target = settings.upload_dir / file.filename
    data = await file.read()
    target.write_bytes(data)
    return target


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def transcribe_media(path: Path) -> tuple[str, list[dict]]:
    transcript_path = path.with_suffix(path.suffix + ".txt")
    if transcript_path.exists():
        text = transcript_path.read_text(encoding="utf-8")
    else:
        text = (
            "Demo transcript for uploaded media. The speaker introduces MediaMind AI, "
            "explains document question answering, then discusses timestamps and playback."
        )
    chunks = split_chunks(text, size=220, overlap=0) or [text]
    segments = []
    for index, chunk in enumerate(chunks):
        segments.append(
            {
                "text": chunk,
                "start_time": float(index * 30),
                "end_time": float(index * 30 + 30),
                "token_set": " ".join(sorted(set(tokenize(chunk)))),
            }
        )
    return text, segments


def build_text_chunks(text: str) -> list[dict]:
    return [
        {
            "text": chunk,
            "start_time": None,
            "end_time": None,
            "token_set": " ".join(sorted(set(tokenize(chunk)))),
        }
        for chunk in split_chunks(text)
    ]


def extract_upload(path: Path, content_type: str) -> tuple[str, str, list[dict]]:
    if content_type == "application/pdf":
        text = extract_pdf_text(path)
        chunks = build_text_chunks(text)
    elif content_type.startswith("audio/") or content_type.startswith("video/"):
        text, chunks = transcribe_media(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = build_text_chunks(text)
    if not text:
        text = "No readable text was extracted from this file."
    return text, summarize_text(text), chunks
