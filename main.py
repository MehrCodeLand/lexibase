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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # accessible externally (e.g., from Docker)
        port=8000,
        reload=True       # auto reload when you edit code
    )

