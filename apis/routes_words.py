from fastapi import APIRouter , Query , HTTPException
from models.schemas import WordItem , SearchQuery
from core.qdrant_client import insert_word , advanced_filter_search , search_similar , hybrid_search , get_all_categories ,search_by_category , update_words_with_categories
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

        insert_word(vector=vector, payload=payload)
        return {"message": f"Word '{item.word}' added successfully."}

    except Exception as e:
        logger.info(f"we have issues in add words api {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search_word")
def search_word(query: SearchQuery):
    try:
        query_vector = get_vector(text=query.word)
        result = search_similar(query_vector, limit=query.limit)
        return [
            {
                "word": r.payload["word"],
                "score": r.score,
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
                "category": r.payload.get("category"),
            }
            for r in result
        ]
    except Exception as e:
        logger.info(f"we have issues in search word {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search_by_meaning")
def search_by_meaning(
    text: str = Query(..., description="Meaning or definition to search"), 
    limit: int = 5
):
    try:
        query_vector = get_vector(text)
        results = search_similar(query_vector, limit)
        return [
            {
                "word": r.payload["word"],
                "score": round(r.score, 3),
                "meaning": r.payload.get("meaning"),
                "synonyms": r.payload.get("synonyms"),
                "category": r.payload.get("category"),
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"Error in search_by_meaning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search_metadata")
def search_metadata(keyword: str, limit: int = 5):
    try:
        from core.qdrant_client import client, COLLECTION_NAME
        filter_ = models.Filter(
            must=[models.FieldCondition(key="meaning", match=models.MatchText(text=keyword))]
        )
        results = client.scroll(
            collection_name=COLLECTION_NAME, 
            scroll_filter=filter_, 
            limit=limit
        )
        points, _ = results
        return [
            {
                "word": p.payload["word"],
                "meaning": p.payload.get("meaning"),
                "synonyms": p.payload.get("synonyms"),
                "category": p.payload.get("category"),
            }
            for p in points
        ]
    except Exception as e:
        logger.error(f"Error in search_metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid_search")
def hybrid_search_endpoint(query: str, keywords: str, limit: int = 5):
    try:
        query_vector = get_vector(query)
        results = hybrid_search(vector=query_vector, keyword=keywords, limit=limit)
        return results
    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/hybrid_search")
def hybrid_search_endpoint(query: str, keywords: str, limit: int = 5):
    try:
        query_vector = get_vector(query)
        results = hybrid_search(vector=query_vector, keyword=keywords, limit=limit)
        return results
    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/categories")
def list_all_categories():
    try:
        categories = get_all_categories()
        return {
            "total_categories": len(categories),
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error in list_all_categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@router.get("/category/{category_name}")
def get_words_in_category(
    category_name: str,
    limit: int = Query(20, description="Number of words to return"),
    offset: int = Query(0, description="Pagination offset")    
):
    try: 
        from core.qdrant_client import client , COLLECTION_NAME
        filter_ = models.Filter(
            must=[models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category_name)
                )]
        )

        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filter_,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        points, next_offset = results

        return {
            "category": category_name,
            "total_returned": len(points),
            "has_more": next_offset is not None,
            "words": [
                {
                    "word": p.payload["word"],
                    "meaning": p.payload.get("meaning"),
                    "synonyms": p.payload.get("synonyms"),
                    "category": p.payload.get("category"),
                }
                for p in points
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_words_in_category: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/search_in_category")
def search_within_category(
    query: str = Query(..., description="Search query text"),
    category: str = Query(..., description="Category name to search within"),
    limit: int = Query(10, description="Number of results")
):
    try: 
        vector = get_vector(query)
        results = search_by_category(
            vector=vector,
            category=category,
            limit=limit
        )

        return {
            "query": query,
            "category": category,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in search_within_category: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/advanced_search")
def advanced_search(
    query: str = Query(None, description="Semantic search query (optional)"),
    category: str = Query(None, description="Filter by category (optional)"),
    meaning_keyword: str = Query(None, description="Filter by meaning keyword (optional)"),
    limit: int = Query(10, description="Number of results")
):
    """
    Advanced search with multiple filters
    Can combine semantic search, category filter, and meaning keyword filter
    At least one parameter (query, category, or meaning_keyword) must be provided
    """
    try:
        if not any([query, category, meaning_keyword]):
            raise HTTPException(
                status_code=400,
                detail="At least one search parameter (query, category, or meaning_keyword) is required"
            )
        
        vector = get_vector(query) if query else None
        
        results = advanced_filter_search(
            vector=vector,
            category=category,
            meaning_keyword=meaning_keyword,
            limit=limit
        )
        
        return {
            "filters": {
                "query": query,
                "category": category,
                "meaning_keyword": meaning_keyword
            },
            "total_results": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced_search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

import json
import os 

@router.get("/category_stats")
def get_category_statistics():
    try:
        categories = get_all_categories()
        
        if not categories:
            return {
                "total_categories": 0,
                "total_categorized_words": 0,
                "average_words_per_category": 0,
                "top_categories": [],
                "statistics": {}
            }
        
        total_words = sum(cat["word_count"] for cat in categories)
        avg_words = total_words / len(categories) if categories else 0
        
        return {
            "total_categories": len(categories),
            "total_categorized_words": total_words,
            "average_words_per_category": round(avg_words, 2),
            "top_categories": categories[:10],
            "statistics": {
                "largest_category": categories[0] if categories else None,
                "smallest_category": categories[-1] if categories else None,
            }
        }
    except Exception as e:
        logger.error(f"Error in get_category_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/cluster_info")
def get_cluster_information():
    """
    Get information about word clusters from the saved JSON file
    Returns cluster metadata and representative words
    """
    try:
        json_path = "word_categories.json"
        
        if not os.path.exists(json_path):
            raise HTTPException(
                status_code=404,
                detail="Cluster information not found. Please run clustering first."
            )
        
        with open(json_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        
        cluster_summary = []
        for category_name, data in clusters.items():
            cluster_summary.append({
                "category": category_name,
                "cluster_id": data.get("cluster_id"),
                "total_words": data.get("total_words"),
                "representative_words": data.get("representative_words", [])[:10],
            })
        
        return {
            "total_clusters": len(clusters),
            "clusters": cluster_summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_cluster_information: {e}")
        raise HTTPException(status_code=500, detail=str(e))
