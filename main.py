from fastapi import FastAPI
from models.schemas import WordItem , SearchQuery
from core.qdrant_client import setup_qdrant
from apis.routes_words import router as word_router

app = FastAPI()

setup_qdrant()

app.include_router(word_router , prefix="/api" , tags=["words"])



@app.get("/health")
def health():
    return {"status" : "ok"}

