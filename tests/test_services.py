from pathlib import Path

from app.database import init_db, store
from app.models import Chunk, Document
from app.services.extraction import build_text_chunks, extract_upload
from app.services import qa
from app.services.qa import answer_question, search_chunks
from app.services.text import keyword_score, split_chunks, summarize_text, tokenize


def test_text_helpers_cover_edges():
    assert tokenize("The FastAPI, AI, and media!") == ["fastapi", "ai", "media"]
    assert split_chunks("", size=5) == []
    assert split_chunks("abcdefghij", size=4, overlap=1) == ["abcd", "defg", "ghij"]
    assert summarize_text("One short sentence.") == "One short sentence."
    long = "Alpha wins. Beta beta wins. Gamma waits. Beta closes."
    assert "Beta" in summarize_text(long, max_sentences=2)
    assert keyword_score("missing", "nothing here") == 0.0
    assert keyword_score("", "nothing here") == 0.0


def test_extraction_fallbacks(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    text, summary, chunks = extract_upload(empty, "text/plain")
    assert text == "No readable text was extracted from this file."
    assert summary == "No readable text was extracted from this file."
    assert chunks == []

    media = tmp_path / "lesson.mp4"
    media.write_bytes(b"video")
    transcript = tmp_path / "lesson.mp4.txt"
    transcript.write_text("Topic one. Topic two playback.", encoding="utf-8")
    text, summary, chunks = extract_upload(media, "video/mp4")
    assert "playback" in text
    assert summary
    assert chunks[0]["start_time"] == 0.0

    assert build_text_chunks("Chunkable text")[0]["start_time"] is None


def test_qa_without_matching_chunks():
    store.reset()
    init_db()
    document = store.add_document(
        Document(
            filename="blank.txt",
            content_type="text/plain",
            file_path="uploads/blank.txt",
            extracted_text="",
            summary="",
        )
    )
    matches = search_chunks(store, "anything", document.id)
    assert matches == []
    answer, citations = answer_question(store, "anything", document.id)
    assert "could not find" in answer
    assert citations == []


def test_qa_uses_document_context_and_local_answer():
    store.reset()
    document = store.add_document(
        Document(
            filename="react.pdf",
            content_type="application/pdf",
            file_path="uploads/react.pdf",
            extracted_text="React is a JavaScript library for building user interfaces. Components are reusable UI blocks.",
            summary="React basics",
        )
    )
    store.add_chunk(
        Chunk(
            document_id=document.id,
            text="React is a JavaScript library for building user interfaces.",
            token_set="react javascript library interfaces",
        )
    )
    answer, citations = answer_question(store, "What is React?", document.id)
    assert "JavaScript library" in answer
    assert citations[0].filename == "react.pdf"


def test_context_from_matches_has_file_and_time_labels():
    store.reset()
    document = store.add_document(
        Document(
            filename="demo.mp3",
            content_type="audio/mpeg",
            file_path="uploads/demo.mp3",
            extracted_text="Audio discusses React hooks.",
            summary="Audio",
        )
    )
    chunk = store.add_chunk(
        Chunk(
            document_id=document.id,
            text="Audio discusses React hooks.",
            start_time=0,
            end_time=30,
            token_set="audio react hooks",
        )
    )
    context = qa._context_from_matches([(chunk, document)])
    assert "File: demo.mp3 | Time: 0s-30s" in context


def test_openai_answer_parses_output_text(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": "React is used for UI."}

    class FakeSettings:
        openai_api_key = "test-key"
        openai_model = "test-model"

    monkeypatch.setattr(qa, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(qa.httpx, "post", lambda *args, **kwargs: FakeResponse())
    assert qa._openai_answer("What is React?", "React context") == "React is used for UI."


def test_openai_answer_parses_nested_output(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": [
                    {"content": [{"type": "output_text", "text": "Nested answer"}]},
                ]
            }

    class FakeSettings:
        openai_api_key = "test-key"
        openai_model = "test-model"

    monkeypatch.setattr(qa, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(qa.httpx, "post", lambda *args, **kwargs: FakeResponse())
    assert qa._openai_answer("Question", "Context") == "Nested answer"


def test_groq_answer_parses_chat_completion(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Groq answer"}}]}

    class FakeSettings:
        groq_api_key = "test-key"
        groq_model = "test-model"
        groq_verify_ssl = True

    monkeypatch.setattr(qa, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(qa.httpx, "post", lambda *args, **kwargs: FakeResponse())
    assert qa._groq_answer("Question", "Context") == "Groq answer"
