import os
from .base_loader import BaseDocumentLoader
from .pdf_loader import PDFLoader
from .txt_loader import TxtLoader

class DocLoaderFactory:
    __loader_registry = {
        ".pdf": PDFLoader,
        ".txt": TxtLoader,
    }

    def get_loader(self, file_path: str) -> BaseDocumentLoader:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext not in self.__loader_registry:
            raise KeyError(f"No loader registered for extension: {ext}")

        return self.__loader_registry[ext]()