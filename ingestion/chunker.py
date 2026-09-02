import hashlib
import uuid
from transformers import AutoTokenizer
from model.schema import Chunk, Document

class TokenChunker:
    def __init__(self, tokenizer_name:str, chunk_size:int = 384, overlap:int = 64): # Define chunking strategy
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document:Document) -> list[Chunk]: # Basic chunking by size
        chunks = [] 
        global_chunk_index = 0
        step_size = self.chunk_size - self.overlap # Skip every n token in loop (i.e. chunk=384 overlap 64 skip every 320 token for each loop cycle)

        for page in document.pages:
            token_ids = self.tokenizer.encode( # Token Embedding
                page.text,
                add_special_tokens=False
            )

            print(
                f"Page {page.page_number}: "
                f"{len(token_ids)} tokens"
            )

            for start in range(0, len(token_ids), step_size): # Loop every 320 token and chunk them
                end = start + self.chunk_size # if start token is 0, then end token is 384
                chunk_tokens = token_ids[start:end] # [0:384], [320:704],.. Skip every 320 tokens but each chunk consists of 384 tokens, this creates a overlap

                if not chunk_tokens:
                    continue

                chunk_text = self.tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True
                )

                chunk_id = self._generate_chunk_id(document_id=document.document_id, page_number=page.page_number, chunk_index=global_chunk_index)

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        filename=document.filename,
                        page_number=page.page_number,
                        chunk_index=global_chunk_index,
                        text=chunk_text.strip()
                    )
                )
                global_chunk_index += 1

        return chunks

    def _generate_chunk_id(self, document_id:str, page_number:int, chunk_index:int) -> str:
        value = (
            f"{document_id}:"
            f"{page_number}:"
            f"{chunk_index}"
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))