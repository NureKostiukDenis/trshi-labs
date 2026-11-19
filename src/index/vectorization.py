import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import QdrantConfig, get_qdrant_client

DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclid": Distance.EUCLID,
    "manhattan": Distance.MANHATTAN,
}

DEFAULT_PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,
    "title": PayloadSchemaType.KEYWORD,
    "section": PayloadSchemaType.KEYWORD,
    "tier": PayloadSchemaType.KEYWORD,
    "categories": PayloadSchemaType.KEYWORD,
    "chapter": PayloadSchemaType.KEYWORD,
    "quest_id": PayloadSchemaType.KEYWORD,
    "is_main": PayloadSchemaType.BOOL,
    "required_items": PayloadSchemaType.KEYWORD,
}


class QdrantVectorizer:
    """Helper for building Qdrant PointStruct objects and managing collection indexes."""

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        client: QdrantClient | None = None,
    ) -> None:
        self.config = qdrant_config
        self.client = client or get_qdrant_client(qdrant_config)
        self.collection_name = qdrant_config.collection_name

    @staticmethod
    def generate_point_id(payload: Mapping[str, Any]) -> str:
        """Generate a deterministic UUID5 from chunk identity fields."""
        identity = "|".join(
            [
                str(payload.get("source", "")),
                str(payload.get("chapter", "")),
                str(payload.get("quest_id", "")),
                str(payload.get("title", "")),
                str(payload.get("section", "")),
                str(payload.get("chunk_index", 0)),
            ]
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    @classmethod
    def build_point(
        cls,
        payload: dict[str, Any],
        vector: list[float],
        point_id: str | None = None,
    ) -> PointStruct:
        """Create a Qdrant PointStruct with a generated or explicit ID."""
        resolved_id = point_id or cls.generate_point_id(payload)
        return PointStruct(id=resolved_id, vector=vector, payload=payload)

    @classmethod
    def build_points(
        cls,
        items: Sequence[tuple[dict[str, Any], list[float]]],
    ) -> list[PointStruct]:
        """Convert (payload, vector) pairs into a list of PointStruct objects."""
        return [cls.build_point(payload, vector) for payload, vector in items]

    def ensure_collection(self, vector_size: int) -> None:
        """Create the Qdrant collection if it does not already exist."""
        if self.client.collection_exists(self.collection_name):
            return
        distance = DISTANCE_MAP.get(self.config.distance.lower(), Distance.COSINE)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def create_payload_indexes(
        self,
        fields: Mapping[str, PayloadSchemaType] | Sequence[str] | None = None,
    ) -> None:
        """Create Qdrant payload indexes for metadata filtering."""
        if fields is None:
            schema_map: Mapping[str, PayloadSchemaType] = DEFAULT_PAYLOAD_INDEXES
        elif isinstance(fields, Mapping):
            schema_map = fields
        else:
            schema_map = {name: PayloadSchemaType.KEYWORD for name in fields}

        for field_name, field_schema in schema_map.items():
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def upsert_points(
        self,
        points: Sequence[PointStruct],
        batch_size: int = 128,
    ) -> int:
        """Upsert PointStruct objects into the collection in batches."""
        total = 0
        for start in range(0, len(points), batch_size):
            batch = list(points[start : start + batch_size])
            self.client.upsert(collection_name=self.collection_name, points=batch)
            total += len(batch)
        return total
