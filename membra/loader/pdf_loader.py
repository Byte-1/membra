import fitz
import asyncio
from typing import List
from .base_loader import BaseDocumentLoader
from ..utils.file_validation import validate_file

class PDFLoader(BaseDocumentLoader):
    async def load(self, path: str) -> List[str]:
        try:
            # Validate if provided path has a valid file
            await validate_file(path)

            # Run PDF loading in executor since it's IO-bound
            loop = asyncio.get_event_loop()
            def load_pdf():
                doc = fitz.open(path)
                pages = [page.get_text() for page in doc]
                doc.close()
                return pages
                
            return await loop.run_in_executor(None, load_pdf)
        except Exception as e:
            raise RuntimeError(f"[PDFLoader] experienced failure due to error: {e}") from e