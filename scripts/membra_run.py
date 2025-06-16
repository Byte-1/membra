import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from membra.loader.loader_factory import DocLoaderFactory
from membra.chunker.chunker_factory import ChunkerFactory
from membra.embedder.embedder_factory import Embedderfactory
from membra.vectorstore.in_memory_vectorstore import InMemoryVectorStore
from membra.retriever.retriever_factory import RetrieverFactory
from membra.cli import app

def load_documents(path: str):
    loader = DocLoaderFactory().get_loader(path)
    pages = loader.load(path)
    print(f"Loaded {len(pages)} pages from {path}")
    return pages

def chunk_documents(pages, method="token", chunk_size=64):
    chunker = ChunkerFactory().get_chunker(method, chunk_size=chunk_size)
    chunks = chunker.chunk(pages)
    print(f"Generated {len(chunks)} chunks using {method} chunker.")
    return chunks

def embed_and_store(project: str, chunks, method="hf"):
    embedder = Embedderfactory().get_embedder(method)
    store = InMemoryVectorStore(embedder)
    store.add(project=project, chunks=chunks)
    store.persist()
    print(f"Stored chunks for project: {project}")
    return store

def run_query(store, project: str, query: str, top_k=3, min_score=0.6):
    retriever = RetrieverFactory().get_retriever("simple", store)
    results = retriever.retrieve(query_text=query, project_id=project, top_k=top_k, min_score=min_score)
    print(f"Query: {query}")
    print(f"Top {top_k} results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")

# def main():
#     project = "test_project"
#     path = "C:/Users/epuru/Downloads/90_day_war_map_planner_puru_vohra.pdf"
#     query = "What is dummy.dumdum?"

#     pages = load_documents(path)
#     chunks = chunk_documents(pages, method="token", chunk_size=64)
#     store = embed_and_store(project, chunks)
#     run_query(store, project, query)

if __name__ == "__main__":
    app()
