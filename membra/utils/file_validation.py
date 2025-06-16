import os
import asyncio

async def validate_file(file_path: str):
    loop = asyncio.get_event_loop()
    
    def check_file():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File at path {file_path}, doesn't exist, provide the correct path")
        if not os.path.isfile(file_path):
            raise ValueError(f"Expected a file but got something else: {file_path}")
        if not os.path.getsize(file_path):
            raise ValueError(f"The file at path: {file_path} is empty")
    
    await loop.run_in_executor(None, check_file)