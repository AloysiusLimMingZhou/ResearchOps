import hashlib
from pathlib import Path
from pypdf import PdfReader
from model.schema import Document, Page

class PdfParser:
    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        reader = PdfReader(path) # Call pypdf library to break down the pdf file
        pages = [] # Create an list  of pages for storing pdf content later

        for page_number, page in enumerate(reader.pages, start=1): # Loop through pdf by pages starting from page 1
            text = page.extract_text() or "" # Extract the text per page of document
            pages.append( # Append page number (index) & its content into the main list, follow Page format given by model.schema
                Page(
                    page_number=page_number,
                    text=text
                )
            )
        document_id = self._generate_document_id(path) # Generate a unique hash for each document

        title = None

        if reader.metadata: # If title exists extract the title from the pdf metadata
            title = reader.metadata.title

        return Document( # Handle the given document after reading it by giving it a unique id, followed by its name, title & text for each page
            document_id=document_id,
            filename=path.name,
            title=title,
            pages=pages
        )

    def _generate_document_id(self, path: Path) -> str: # Generate a unique hash of the document with SHA 256
        hasher = hashlib.sha256()

        with path.open('rb') as file:
            while chunk := file.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()