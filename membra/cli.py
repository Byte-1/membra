import typer
from membra.pipeline.ingestion_pipeline import ingest_document
from membra.pipeline.query_pipeline import query_pipeline

app = typer.Typer()

@app.command()
def ingest(
    file: str = typer.Option(..., "--file", "-f", help="Path to the input file (PDF or TXT)"),
    project_id: str = typer.Option(..., "--project_id", "-p", help="Unique identifier for the project"),
    chunker_type: str = typer.Option("sentence", "--chunker_type", "-ct", help="Chunking strategy: simple, sentence, or token"),
    embedder_type: str = typer.Option("hf", "--embedder_type", "-et", help="Embedding model: hf or openai"),
    chunk_size: int = typer.Option(5, "--chunk_size", "-cs", help="Number of words/tokens/sentences per chunk"),
    overlap: int = typer.Option(1, "--overlap", "-o", help="Number of overlapping elements between chunks")
):
    """
    Ingest a document into Membra (load → chunk → embed → store).
    """
    import asyncio
    asyncio.run(ingest_document(
        path=file,
        project_id=project_id,
        chunker_type=chunker_type,
        embedder_type=embedder_type,
        chunk_size=chunk_size,
        overlap=overlap
    ))

@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Question to ask about the ingested document"),
    project_id: str = typer.Option(..., "--project_id", "-p", help="Project identifier used during ingestion"),
    top_k: int = typer.Option(5, "--top_k", "-k", help="Number of top relevant chunks to return"),
    min_score: float = typer.Option(0.0, "--min_score", "-ms", help="Minimum cosine similarity threshold"),
    llm_mode: str = typer.Option("local", "--llm_mode", "-llm_host", help="Which LLM hosting you want to use local/online"),
    llm_name: str = typer.Option("llama3", "--llm_name", "-llm", help="Which LLM offering you want to use openai, local, ollama, etc.")
):
    """
    Query an ingested document project for relevant context chunks.
    """
    import asyncio
    asyncio.run(query_pipeline(question, project_id, top_k, min_score, llm_mode, llm_name))

