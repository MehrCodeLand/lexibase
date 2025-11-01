from fastapi import APIRouter , Query
from models.schemas import WordItem , SearchQuery
from core.qdrant_client import insert_word , search_similar , hybrid_search
from core.embeddings import get_vector
from loguru import logger
from qdrant_client.http import models


router = APIRouter()

@router.post("/add_words")
def add_words(item: WordItem):
    try:

        vector = get_vector(text=item.word)
        payload = {
            "word": item.word,
            "meaning": item.meaning,
            "synonyms": item.synonyms,
            "antonyms": item.antonyms,
            "examples": item.examples
        }

        insert_word(vector=vector , payload=payload)
        return {"message": f"Word '{item.word}' added successfully."}

    except Exception as e :
        logger.info(f" we have issues in add words api {e}")


@router.post("/search_word")
def search_word(query : SearchQuery):
    try:
        query_vector = get_vector(text=query.word)
        result = search_similar(query_vector , limit=query.limit)
        return [
            {
                "word": r.payload["word"],
                "score": r.score,
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
            }
            for r in result
        ]
    except Exception as e :
        logger.info(f"we have issues in search word {e}")
        return None 


@router.get("/search_by_meaning")
def search_by_meaning(text: str = Query(..., description="Meaning or definition to search"), limit: int = 5):
    query_vector = get_vector(text)
    results = search_similar(query_vector, limit)
    return [
        {
            "word": r.payload["word"],
            "score": round(r.score, 3),
            "meaning": r.payload.get("meaning"),
            "synonyms": r.payload.get("synonyms"),
        }
        for r in results
    ]


@router.get("/search_metadata")
def search_metadata(keyword: str, limit: int = 5):
    # basic text filter using Qdrant’s metadata filtering
    from core.qdrant_client import client, COLLECTION_NAME
    filter_ = models.Filter(
        must=[models.FieldCondition(key="meaning", match=models.MatchText(text=keyword))]
    )
    results = client.scroll(collection_name=COLLECTION_NAME, scroll_filter=filter_, limit=limit)
    points, _ = results
    return [
        {
            "word": p.payload["word"],
            "meaning": p.payload.get("meaning"),
            "synonyms": p.payload.get("synonyms"),
        }
        for p in points
    ]


@router.get("/hybrid_search")
def hybrid_search(query: str , keywords : str , limit : int = 5 ):
    results = hybrid_search(query=query , keywords=keywords , limit=limit)
    return results


