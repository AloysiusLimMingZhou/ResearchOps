from pydantic import BaseModel

class Page(BaseModel): # Provide a schema format for each document page. Each page should consists of a page number and content
    page_number: int
    text: str

class Document(BaseModel): # Provide a schema format for document. Each document should have a unique hash id, name, title & content using Page schema which consists of page number and text per page
    document_id: str
    filename: str
    title: str | None = None
    pages: list[Page]

class Chunk(BaseModel): # Provide a schema format for extracted chunks. Each chunk should have a unique id, followed by the document id for referencing, filename, document page, and content
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str

class SearchResults(BaseModel): # Provide a schema format for search results, which takes chunk id as reference, followed by the content of the chunk, and an additional score which shows the likelihood of the chunk being the retrieved answers
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float