from app.config import settings
from qdrant_client import QdrantClient

def get_qdrant_client():
    if settings.QDRANT_ENDPOINT:
        return QdrantClient(url=settings.QDRANT_ENDPOINT, api_key=settings.QDRANT_API_KEY)
    else:
        host = "localhost"
        port = 6333
        return QdrantClient(host=host, port=port, api_key=settings.QDRANT_API_KEY)

def get_existing_ids(client: QdrantClient, collection_name: str) -> set:
    existing_ids = set()
    offset = None
    limit = 100  

    while True:
        response = client.scroll(collection_name=collection_name, limit=limit, offset=offset)
        result = response.result if hasattr(response, "result") else response.get("result", {})
        points = result.get("points", [])
        if not points:
            break
        for point in points:
            existing_ids.add(point["id"])
        offset = result.get("next_page_offset")
        if not offset:
            break

    return existing_ids

def upsert_points(client: QdrantClient, collection_name: str, points: list):
    return client.upsert(collection_name=collection_name, points=points)
