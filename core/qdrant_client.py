from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import os
from loguru import logger
from typing import List


logger.add("qdrant_clint.txt")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "words"

client = QdrantClient(QDRANT_URL)


def setup_qdrant(size: int = 384):
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not collection_exists:
            # Only create if it doesn't exist
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection '{COLLECTION_NAME}' created successfully")
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists, skipping creation")
    
    except Exception as e:
        logger.error(f"Issues in setup_qdrant: {e}")


    
def insert_word(vector, payload):
    try: 
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            ]
        )
    except Exception as e :
        logger.info(f"we have issues in insert_word {e}")


def search_similar(vector, limit=3):
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit
        )
    except Exception as e :
        logger.info(f"issues in search {e}")
        return None


def hybrid_search(vector , keyword : str , limit : int = 5 )-> List[dict]:
    try: 
        filter_ = models.Filter(
            must=[models.FieldCondition(key="meaning" , match=models.MatchText(text=keyword))]
        )

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=filter_,
            limit=limit
        )

        return [
            {
                "word": r.payload["word"],
                "score": r.score,
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
            }
            for r in results
        ]
    except Exception as e :
        logger.info(f"we have issues here {e}")
        return None

        