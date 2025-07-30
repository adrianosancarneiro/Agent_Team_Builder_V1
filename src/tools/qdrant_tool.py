import os
from typing import List, Any, Optional
import requests  # using requests for simplicity; could use qdrant-client library

class QdrantTool:
    """Minimal client for Qdrant vector database operations."""
    def __init__(self):
        # Qdrant URL could be configured via env or config
        self.base_url = os.getenv("QDRANT_URL", "http://192.168.0.83:6333")
        # Optional: collection name could be tenant-specific
        self.collection = os.getenv("QDRANT_COLLECTION", "agent_vectors")

    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[dict] = None) -> List[Any]:
        """Search the Qdrant vector collection for nearest vectors to the query embedding."""
        url = f"{self.base_url}/collections/{self.collection}/points/search"
        payload = {
            "vector": query_embedding,
            "limit": top_k
        }
        if filters:
            payload["filter"] = filters
        try:
            res = requests.post(url, json=payload, timeout=5)
            res.raise_for_status()
            results = res.json().get("result", [])
            return results  # Each result contains e.g. an "id" and "score" and possibly payload
        except Exception as e:
            # In a real system, handle exceptions and logging appropriately
            print(f"Qdrant search error: {e}")
            return []

    def upsert(self, points: List[dict]) -> bool:
        """Insert or update points (vectors with payload) into the collection."""
        url = f"{self.base_url}/collections/{self.collection}/points"
        try:
            res = requests.put(url, json={"points": points}, timeout=5)
            res.raise_for_status()
            return True
        except Exception as e:
            print(f"Qdrant upsert error: {e}")
            return False
