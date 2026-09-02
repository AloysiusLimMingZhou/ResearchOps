import argparse
from embeddings.embedder import Embedder
from evaluation.retrieval_eval import evaluate
from ingestion.chunker import TokenChunker
from ingestion.cleaner import TextCleaner
from ingestion.parser import PdfParser
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def build_components():
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    vector_store = VectorStore(collection_name="researchops")
    vector_store.create_collection(vector_size=embedder.dimension) # Make sure vector DB has the same dimension size as the embedding model (i.e. 784 = 784)
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    return embedder, vector_store, retriever

def ingest(file_path:str):
    embedder, vector_store, _ = build_components()
    parser = PdfParser()
    cleaner = TextCleaner()
    chunker = TokenChunker(
        tokenizer_name=EMBEDDING_MODEL,
        chunk_size=384,
        overlap=64
    )

    # 1. Parse PDF Document
    document = parser.parse(file_path=file_path)
    print(f"Parsed {len(document.pages)} pages")

    # 2. Clean Parsed Document
    document = cleaner.clean_document(document=document)
    print("Cleaned Document")

    # 3. Chunk Cleaned Document
    chunks = chunker.chunk_document(document=document)
    print(f"Created {len(chunks)} chunks")

    # 4. Embed the chunks
    vectors = embedder.embed_documents(
        [chunk.text for chunk in chunks]
    )
    print(f"Generated {len(vectors)} embeddings")

    # 5. Store
    vector_store.upsert(chunks=chunks, vectors=vectors)
    print(f"Stored {len(chunks)} of chunks into Qdrant")

def search(query:str):
    _, _, retriever = build_components()
    results = retriever.retrieve(query=query, top_k=5) # Retrieve top 5 results most relevant to the question from user query

    for rank, result in enumerate(results, start=1):
        print()
        print("=" * 60)
        print( # Print Chunk Rank & Score. Score here means most relevant text to query
            f"#{rank} | "
            f"Score: {result.score:.4f}"
        )

        print( # Print filename & page number of the chunks
            f"{result.filename} | "
            f"Page {result.page_number}"
        )
        print()

        print(result.text) # Print Chunk Text

def run_evaluation(): # Evaluate using our eval formulas
    _, _, retriever = build_components()
    evaluate(retriever=retriever, dataset_path="evaluation/dataset.json")

def main(): # Add arguments to the main functions
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparser.add_parser("ingest")
    ingest_parser.add_argument("file")

    search_parser = subparser.add_parser("search")
    search_parser.add_argument("query")

    subparser.add_parser("eval")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.file)

    if args.command == "search":
        search(args.query)

    if args.command == "eval":
        run_evaluation()

if __name__ == "__main__":
    main()