from fastembed import TextEmbedding

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_vector(text: str):
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()