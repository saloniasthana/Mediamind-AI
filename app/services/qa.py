import logging
import re

import httpx

from app.database import MongoStore
from app.schemas import Citation
from app.config import get_settings
from app.services.text import keyword_score, summarize_text, tokenize


MAX_CONTEXT_CHARS = 16000
logger = logging.getLogger(__name__)


def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a document Q&A assistant. Answer only from the uploaded file context. "
        "If the context does not contain the answer, say that the uploaded file does not include enough information. "
        "Give a clear, useful answer in simple language.\n\n"
        f"Question:\n{question}\n\nUploaded file context:\n{context}"
    )


def _context_from_matches(matches) -> str:
    parts = []
    seen = set()
    for chunk, document in matches:
        key = (document.id, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        label = f"File: {document.filename}"
        if chunk.start_time is not None:
            label += f" | Time: {chunk.start_time:.0f}s-{chunk.end_time:.0f}s"
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _context_from_document(db: MongoStore, document_id: int | None, matches) -> str:
    if not document_id:
        return _context_from_matches(matches)
    document = db.get_document(document_id)
    if not document:
        return _context_from_matches(matches)
    text = document.extracted_text.strip()
    if not text:
        return _context_from_matches(matches)
    return f"File: {document.filename}\n{text[:MAX_CONTEXT_CHARS]}"


def _local_answer(question: str, context: str) -> str:
    context_without_labels = re.sub(r"File: [^\n]+", "", context)
    cleaned = re.sub(r"\s+", " ", context_without_labels)
    sentences = [line.strip() for line in re.split(r"(?<=[.!?])\s+", cleaned) if line.strip()]
    query_terms = set(tokenize(question))
    if not sentences:
        return "I could not find relevant content in the uploaded files."

    question_lower = question.lower()
    if question_lower.startswith(("what is", "what are", "define", "explain")):
        for sentence in sentences:
            lower = sentence.lower()
            if query_terms and any(term in lower for term in query_terms):
                next_index = sentences.index(sentence) + 1
                extra = sentences[next_index] if next_index < len(sentences) else ""
                answer = sentence if not extra else f"{sentence} {extra}"
                return "Based on the uploaded content: " + answer

    useful = []
    for sentence in sentences:
        lower = sentence.lower()
        if query_terms and any(term in lower for term in query_terms):
            if not lower.startswith("file:"):
                useful.append(sentence)
    if useful:
        return "Based on the uploaded content: " + ". ".join(useful[:5]) + "."
    summary = summarize_text(context, max_sentences=4)
    return f"Based on the uploaded content: {summary}" if summary else "I could not find relevant content in the uploaded files."


def _openai_answer(question: str, context: str) -> str | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openai_model,
            "input": _build_prompt(question, context),
            "max_output_tokens": 700,
        },
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("output_text"):
        return data["output_text"].strip()
    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return "\n".join(texts).strip() or None


def _groq_answer(question: str, context: str) -> str | None:
    settings = get_settings()
    if not settings.groq_api_key:
        return None

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using only the uploaded document context.",
                },
                {"role": "user", "content": _build_prompt(question, context)},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        },
        timeout=45,
        verify=settings.groq_verify_ssl,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content", "").strip() or None


def search_chunks(db: MongoStore, question: str, document_id: int | None = None, limit: int = 4):
    rows = db.list_chunks_with_documents(document_id)
    scored = [
        (keyword_score(question, chunk.text + " " + chunk.token_set), chunk, document)
        for chunk, document in rows
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, document) for score, chunk, document in scored[:limit] if score > 0] or [
        (chunk, document) for _, chunk, document in scored[:limit]
    ]


def answer_question(db: MongoStore, question: str, document_id: int | None = None):
    matches = search_chunks(db, question, document_id, limit=8)
    context = _context_from_document(db, document_id, matches)
    if context.strip():
        try:
            answer = _groq_answer(question, context) or _openai_answer(question, context) or _local_answer(question, context)
        except Exception as exc:
            logger.warning("AI provider failed; using local fallback: %s", exc)
            answer = _local_answer(question, context)
    else:
        answer = "I could not find relevant content in the uploaded files."
    citations = [
        Citation(
            document_id=document.id,
            filename=document.filename,
            text=chunk.text,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )
        for chunk, document in matches
    ]
    return answer, citations
