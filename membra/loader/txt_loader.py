from typing import List
from .base_loader import BaseDocumentLoader
from ..utils.file_validation import validate_file

class TxtLoader(BaseDocumentLoader):
    def load(self, path: str) -> List[str]:
        try: 
            # Validate if provided path has a valid file
            validate_file(path)
            with open(path, "r", encoding="utf-8") as f:
                return [f.read()]
        except Exception as e:
            raise RuntimeError(f"[TxtLoader] experienced failure due to error: {e}") from e