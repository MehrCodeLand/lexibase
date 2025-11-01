from fastapi import APIRouter
from models.schemas import WordItem , SearchQuery
from core.qdrant_client import insert_word , search_similar
from core.embeddings import get_vector
from loguru import logger

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
