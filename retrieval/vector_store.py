from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from model.schema import Chunk

class VectorStore:
    def __init__(self, collection_name:str = "researchops", url:str = "http://localhost:6333"): # Declare Qdrant Vector DB Instance
        self.client = QdrantClient(url=url)
        self.collection_name=collection_name

    def create_collection(self, vector_size:int) -> None:
        if self.client.collection_exists(self.collection_name): return # If collection, which is table in vector database term exist, then skip this process

        self.client.create_collection( # Else, create a collection which is a SQL table in vector db term and use cosine distance between the embeddings
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE) # Vector size is the size of Vector DB, usually we'd like it to be the same as embedding dimension. Cosine distance is used to compare query vector & stored vector
        )

    def upsert(self, chunks:list[Chunk], vectors:list[list[float]]): # Upsert is SQL INSERT chunks into the vector db where points represents SQL rows in vector db
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(PointStruct( # PointStruct is like SQL Schema (i.e. for SQL table u have primary key, columns, etc) but for vector db. It has point (row) id, vector (data), payload (metadata)
                id=chunk.chunk_id,
                vector=vector,
                payload={ # JSON Metadata
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text
                }
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit:int = 5): # Basically SQL SELECT * FROM TABLE WHERE... LIMIT = 5;
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True # Metadata Filter, include metadata basically
        )

        return result.points