from embeddings.embedder import Embedder
from model.schema import SearchResults
from retrieval.vector_store import VectorStore

class Retriever:
    def __init__(self, embedder:Embedder, vector_store:VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query:str, top_k:int = 5):
        query_vector = self.embedder.embed_query(query) # Embed the user query
        points = self.vector_store.search( # Search for relevant embeddings from vector db content and limit to top 5 relevant results based on embedding score. Closer words with similar semantic meaning will get higher score
            query_vector=query_vector,
            limit=top_k
        )

        results = []
        for point in points:
            payload = point.payload # Retrieve chunk_id, document_id, chunk_text, etc
            results.append( # Append results from point & payload
                SearchResults(
                    chunk_id=str(point.id), # point.id = chunk.chunk_id from vector_store.py
                    document_id=payload["document_id"],
                    filename=payload["document_id"],
                    page_number=payload["page_number"],
                    text=payload["text"],
                    score=point.score
                )
            )

        return results