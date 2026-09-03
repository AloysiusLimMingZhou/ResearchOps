import re
from model.schema import Document, Page

class TextCleaner:
    def clean_text(self, text: str) -> str:
        # Windows / Mac line endings -> Unix
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Non-breaking space
        text = text.replace("\xa0", " ") # Convert U+00A0 non-breaking space in pdf -> U+0020 normal breaks

        # Remove spaces before newline
        text = re.sub(r"[ \t]+\n", "\n", text) # i.e. "hello       \nworld" -> "hello\nworld"

        # Collapse repeated spaces
        text = re.sub(r"[ \t]{2,}", " ", text) # i.e. "The     Transformer     architecture" -> "The Transformer architecture"

        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text) # i.e. "Paragraph A\n\n\nParagraph B" -> "Paragraph A\nParagraph B"

        return text.strip()

    def clean_document(self, document: Document) -> Document:
        cleaned_pages = []

        for page in document.pages: # Apply clean_text function to the content of each page
            cleaned_pages.append(
                Page(
                    page_number=page.page_number,
                    text=self.clean_text(page.text)
                )
            )

        return document.model_copy( # Create a copy of the document, and update the document with the cleaned text per page
            update={
                "pages": cleaned_pages
            }
        )