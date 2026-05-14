from collections.abc import Generator

from pymongo import ASCENDING, MongoClient, ReturnDocument

from app.config import get_settings
from app.models import Chunk, Document


class MongoStore:
    def __init__(self, client, database_name: str):
        self.client = client
        self.db = client[database_name]
        self.documents = self.db["documents"]
        self.chunks = self.db["chunks"]
        self.counters = self.db["counters"]

    def init(self) -> None:
        self.documents.create_index([("id", ASCENDING)], unique=True)
        self.documents.create_index([("created_at", ASCENDING)])
        self.chunks.create_index([("id", ASCENDING)], unique=True)
        self.chunks.create_index([("document_id", ASCENDING)])

    def reset(self) -> None:
        self.documents.delete_many({})
        self.chunks.delete_many({})
        self.counters.delete_many({})

    def _next_id(self, name: str) -> int:
        counter = self.counters.find_one_and_update(
            {"_id": name},
            {"$inc": {"value": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(counter["value"])

    def add_document(self, document: Document) -> Document:
        data = document.model_dump()
        data["id"] = self._next_id("documents")
        self.documents.insert_one(data)
        return Document(**data)

    def add_chunk(self, chunk: Chunk) -> Chunk:
        data = chunk.model_dump()
        data["id"] = self._next_id("chunks")
        self.chunks.insert_one(data)
        return Chunk(**data)

    def list_documents(self) -> list[Document]:
        rows = self.documents.find({}, {"_id": 0}).sort("created_at", -1)
        return [Document(**row) for row in rows]

    def get_document(self, document_id: int) -> Document | None:
        row = self.documents.find_one({"id": document_id}, {"_id": 0})
        return Document(**row) if row else None

    def list_chunks_with_documents(self, document_id: int | None = None) -> list[tuple[Chunk, Document]]:
        query = {"document_id": document_id} if document_id else {}
        rows = self.chunks.find(query, {"_id": 0})
        pairs: list[tuple[Chunk, Document]] = []
        for row in rows:
            chunk = Chunk(**row)
            document = self.get_document(chunk.document_id)
            if document:
                pairs.append((chunk, document))
        return pairs


def _create_client():
    settings = get_settings()
    if settings.mongodb_url.startswith("mongomock://"):
        import mongomock

        return mongomock.MongoClient()
    return MongoClient(settings.mongodb_url)


store = MongoStore(_create_client(), get_settings().mongodb_database)


def init_db() -> None:
    store.init()


def get_database() -> Generator[MongoStore, None, None]:
    yield store
