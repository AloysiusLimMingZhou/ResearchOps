from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name:str = "BAAI/bge-small-en-v1.5"): # Define embedding model
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int: # Get the embedding dimension of the model (i.e. 784, 384, 1024, 2048,...)
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]: # Embed the documents
        embeddings = self.model.encode_document(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query:str) -> list[float]: # Embed the queries from user
        embeddings = self.model.encode_query(query, normalize_embeddings=True)
        print(self.model.prompts)
        return embeddings.tolist()