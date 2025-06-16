from ..loader.loader_factory import DocLoaderFactory
from ..chunker.chunker_factory import ChunkerFactory
from ..embedder.embedder_factory import Embedderfactory
from ..vectorstore.vectorstore_factory import VectorStoreFactory
import os, time

async def ingest_document(path: str, project_id: str, chunker_type: str = "sentence", embedder_type: str = "hf", chunk_size: int = 500, overlap: int = None, store_name:str = "faiss") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Ingestion] File not found at path: {path}")

    start_time = time.perf_counter()
    print(f"[Ingestion] Loading document from: {path}")
    loader = DocLoaderFactory().get_loader(path)
    texts = await loader.load(path)
    print(f"[Ingestion] Loaded {len(texts)} pages")
    end_time = time.perf_counter()
    print(f"[Ingestion-Loading] Took {end_time - start_time:.2f} seconds")

    start_time = time.perf_counter()
    print(f"[Ingestion] Chunking text using: {chunker_type}")
    chunker = ChunkerFactory().get_chunker(chunker_type, chunk_size=chunk_size, overlap=overlap)
    chunks = await chunker.chunk(texts)
    print(f"[Ingestion] Generated {len(chunks)} chunks")
    end_time = time.perf_counter()
    print(f"[Ingestion-Chunking] Took {end_time - start_time:.2f} seconds")


    start_time = time.perf_counter()
    print(f"[Ingestion] Embedding using: {embedder_type}")
    embedder = Embedderfactory().get_embedder(embedder_type)
    vectorstore = await VectorStoreFactory.get_vectorstore(store_name, embedder)

    print(f"[Ingestion] Storing chunks under project: {project_id}")
    await vectorstore.add(project=project_id, chunks=chunks)

    print(f"[Ingestion] Persisting vectorstore to disk")
    await vectorstore.persist()

    print("[Ingestion] Document ingestion complete.")

    end_time = time.perf_counter()
    print(f"[Ingestion-Vectorization] Took {end_time - start_time:.2f} seconds")
