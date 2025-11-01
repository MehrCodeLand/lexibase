from pydantic import BaseModel

class WordItem(BaseModel):
    word: str
    meaning: str
    synonyms: list[str] = []
    antonyms: list[str] = []
    examples: list[str] = []

class SearchQuery(BaseModel):
    word: str
    limit: int = 3