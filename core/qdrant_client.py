from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import os
from loguru import logger
from typing import List, Dict

logger.add("qdrant_clint.txt")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "words"

client = QdrantClient(QDRANT_URL)


def setup_qdrant(size: int = 384):
    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not collection_exists:
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
    except Exception as e:
        logger.info(f"we have issues in insert_word {e}")


def search_similar(vector, limit=3):
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit
        )
    except Exception as e:
        logger.info(f"issues in search {e}")
        return None


def hybrid_search(vector, keyword: str, limit: int = 5) -> List[dict]:
    try: 
        filter_ = models.Filter(
            must=[models.FieldCondition(key="meaning", match=models.MatchText(text=keyword))]
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
    except Exception as e:
        logger.info(f"we have issues here {e}")
        return None


# Updates all words in Qdrant with their assigned category labels
# Takes categorized clusters dict and adds category field to each word's payload
# Uses batch operations for efficiency when updating large numbers of words
def update_words_with_categories(categorized_clusters: Dict) -> Dict[str, int]:
    from tqdm import tqdm
    
    stats = {
        "total_updated": 0,
        "total_failed": 0,
        "categories_processed": 0
    }
    
    logger.info("Starting category update in Qdrant...")
    
    for category_name, data in tqdm(categorized_clusters.items(), desc="Updating categories in Qdrant"):
        stats["categories_processed"] += 1
        words_in_category = data['all_words']
        
        for word in words_in_category:
            try:
                results = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="word",
                            match=models.MatchValue(value=word)
                        )]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, _ = results
                
                if points:
                    point = points[0]
                    client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            "category": category_name,
                            "cluster_id": data['cluster_id']
                        },
                        points=[point.id]
                    )
                    stats["total_updated"] += 1
                else:
                    logger.warning(f"Word '{word}' not found in Qdrant")
                    stats["total_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to update word '{word}': {e}")
                stats["total_failed"] += 1
    
    logger.info(f"Category update completed: {stats['total_updated']} updated, {stats['total_failed']} failed")
    return stats


# Searches for words within a specific category using metadata filtering
# Combines semantic vector search with category filter for precise results
# Returns words that match both the query meaning and belong to specified category
def search_by_category(vector, category: str, limit: int = 10) -> List[dict]:
    try:
        filter_ = models.Filter(
            must=[models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category)
            )]
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
                "score": round(r.score, 3),
                "category": r.payload.get("category"),
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
            }
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Issues in search_by_category: {e}")
        return []


# Retrieves all unique categories stored in Qdrant database
# Uses scroll with distinct values to get list of all category names
# Returns list of category names with their word counts
def get_all_categories() -> List[Dict[str, any]]:
    try:
        all_categories = {}
        
        offset = None
        while True:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["category"],
                with_vectors=False
            )
            
            points, next_offset = results
            
            if not points:
                break
            
            for point in points:
                category = point.payload.get("category")
                if category:
                    if category not in all_categories:
                        all_categories[category] = 0
                    all_categories[category] += 1
            
            offset = next_offset
            if next_offset is None:
                break
        
        category_list = [
            {"category": cat, "word_count": count}
            for cat, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return category_list
        
    except Exception as e:
        logger.error(f"Issues in get_all_categories: {e}")
        return []


# Filters and searches words by multiple criteria
# Supports filtering by category, meaning keywords, and semantic similarity
# Useful for complex queries that need both metadata and vector search
def advanced_filter_search(
    vector=None,
    category: str = None,
    meaning_keyword: str = None,
    limit: int = 10
) -> List[dict]:
    try:
        must_conditions = []
        
        if category:
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            )
        
        if meaning_keyword:
            must_conditions.append(
                models.FieldCondition(
                    key="meaning",
                    match=models.MatchText(text=meaning_keyword)
                )
            )
        
        filter_ = models.Filter(must=must_conditions) if must_conditions else None
        
        if vector is not None:
            results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                query_filter=filter_,
                limit=limit
            )
        else:
            results = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=filter_,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            points, _ = results
            return [
                {
                    "word": p.payload["word"],
                    "category": p.payload.get("category"),
                    "meaning": p.payload.get("meaning"),
                    "synonyms": p.payload.get("synonyms"),
                }
                for p in points
            ]
        
        return [
            {
                "word": r.payload["word"],
                "score": round(r.score, 3),
                "category": r.payload.get("category"),
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
            }
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Issues in advanced_filter_search: {e}")
        return []