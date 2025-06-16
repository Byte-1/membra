# Membra

Membra is a flexible Retrieval-Augmented Generation (RAG) framework that enables efficient document processing, chunking, embedding, and querying with support for both local and cloud-based LLMs.

## Features

- 📄 Multi-format document ingestion (PDF, TXT)
- 🔍 Smart document chunking strategies (sentence-based, token-based, or simple)
- 🧮 Efficient vector embeddings using HuggingFace models
- 💾 FAISS vector store for fast similarity search
- 🤖 Support for both local (Ollama) and cloud LLMs
- ⚡ Parallel processing for document summarization
- 🎯 Contextual query responses with relevant document snippets

## Prerequisites

1. Python 3.9 or higher
2. Local LLM setup (recommended):
   - [Ollama](https://ollama.ai/) installed
   - Required models pulled:
     ```bash
     ollama pull llama2  # or your preferred model
     ```
3. Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Document Ingestion

Ingest documents into the vector store:

```bash
python -m membra_run ingest \
  --file "path/to/your/document.pdf" \
  --project_id "unique-project-name" \
  --chunk_size 30 \  # words per chunk
  --overlap 10 \     # words overlap between chunks
  --chunker_type "sentence"  # sentence, token, or simple
```

### Querying Documents

Query your ingested documents:

```bash
python -m membra_run query \
  --question "Your question here?" \
  --project_id "unique-project-name" \
  --top_k 5 \        # number of relevant chunks to retrieve
  --min_score 0.0 \  # minimum similarity score (0-1)
  --llm_mode "local" \  # local or online
  --llm_name "llama3"   # model name (llama3, phi, etc.)
```

## Architecture

1. **Ingestion Pipeline**:
   - Document Loading → Chunking → Embedding → Vector Storage

2. **Query Pipeline**:
   - Question → Vector Search → Chunk Retrieval → Parallel Summarization → LLM Response

## Configuration

### Supported LLM Models

1. Local Models (via Ollama):
   - llama3 (default)
   - phi
   - Custom models (configured in Ollama)

2. Online Models:
   - OpenAI (requires API key)

### Chunking Strategies

- `sentence`: Split by sentences (recommended for most texts)
- `token`: Split by token count
- `simple`: Simple text splitting

### Vector Store

Currently supports:
- FAISS (default, efficient for local storage)
- In-memory store (for testing)

## Performance Tips

1. Use sentence-based chunking for natural text boundaries
2. Adjust chunk size based on your document type:
   - Technical docs: 30-50 words
   - Narrative text: 50-100 words
3. For faster processing:
   - Use local models via Ollama
   - Enable GPU acceleration if available
   - Adjust parallel processing parameters in config

## Contributing

Contributions are welcome! Please check our contribution guidelines and submit PRs.

