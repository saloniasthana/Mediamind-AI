from pathlib import Path

from fastapi.testclient import TestClient

from app.database import init_db, store
from app.main import app
from app.models import Document


init_db()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_upload_chat_and_documents(tmp_path: Path):
    store.reset()
    content = (
        b"MediaMind AI supports semantic document question answering. "
        b"It summarizes uploaded material and helps users find timestamps."
    )
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    document = response.json()
    assert document["filename"] == "notes.txt"
    assert "semantic" in document["summary"]

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    assert any(item["id"] == document["id"] for item in list_response.json())

    chat_response = client.post(
        "/api/chat",
        json={"question": "What does MediaMind support?", "document_id": document["id"]},
    )
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert "uploaded content" in payload["answer"]
    assert payload["citations"][0]["document_id"] == document["id"]


def test_media_timestamps_and_file_response():
    store.reset()
    response = client.post(
        "/api/upload",
        files={"file": ("demo.mp3", b"fake audio bytes", "audio/mpeg")},
    )
    assert response.status_code == 200
    document = response.json()

    ts_response = client.post(
        "/api/timestamps",
        json={"topic": "timestamps playback", "document_id": document["id"]},
    )
    assert ts_response.status_code == 200
    timestamps = ts_response.json()
    assert timestamps
    assert timestamps[0]["start_time"] == 0.0

    media_response = client.get(f"/api/media/{document['id']}")
    assert media_response.status_code == 200
    assert media_response.content == b"fake audio bytes"


def test_missing_document_media_returns_404():
    store.reset()
    response = client.get("/api/media/999999")
    assert response.status_code == 404


def test_upload_requires_content_type():
    store.reset()
    response = client.post("/api/upload", files={"file": ("bad.bin", b"data", "")})
    assert response.status_code == 400


def test_missing_media_file_returns_404():
    store.reset()
    document = store.add_document(
        Document(
            filename="missing.mp4",
            content_type="video/mp4",
            file_path="uploads/does-not-exist.mp4",
            extracted_text="",
            summary="",
        )
    )
    document_id = document.id
    response = client.get(f"/api/media/{document_id}")
    assert response.status_code == 404
