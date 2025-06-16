from membra.embedder.embedder_factory import Embedderfactory
from membra.vectorstore.vectorstore_factory import VectorStoreFactory
from membra.retriever.retriever_factory import RetrieverFactory
from membra.prompt.prompt_augmenter import PromptAugmenter
from membra.llmconnector.llmconnector_factory import LLMFactory
from membra.summarizer.simple_summarizer import Summarizer
from rich import print
import time
import asyncio

async def query_pipeline(question: str, project_id: str, top_k: int = 5, min_score: float = 0.0, llm_mode="local", llm_name="llama3", store_name:str = "faiss"):
    try:
        # Step 1: Load embedder and vector store
        start_time = time.perf_counter()
        embedder = Embedderfactory().get_embedder("hf")  # You can allow this to be CLI param if needed
        vectorstore = await VectorStoreFactory.get_vectorstore(store_name, embedder)
        end_time = time.perf_counter()
        print(f"[Embedding and Vector Loading] Took {end_time - start_time:.2f} seconds")

        # Step 2: Set up retriever
        start_time = time.perf_counter()
        retriever = RetrieverFactory().get_retriever("simple", vectorstore)

        # Step 3: Retrieve top-k chunks
        results = await retriever.retrieve(
            query_text=question,
            project_id=project_id,
            top_k=top_k,
            min_score=min_score
        )
        end_time = time.perf_counter()
        print(f"[Retrieving Top K chunks] Took {end_time - start_time:.2f} seconds")

        if not results:
            print(f"\n No relevant context found for your query: '{question}'\n")
            return

        print(f"\n🔍 [bold blue]Your Question:[/bold blue] {question}")
        print(f"Top {len(results)} context chunks retrieved:\n")
        for idx, res in enumerate(results):
            print(f"{idx}. {res}\n")
        
        # Step 4. Fetch the LLM and augmenter
        start_time = time.perf_counter()
        llm_factory = LLMFactory.get_llm_factory(llm_mode)
        llm = llm_factory.get_llm(llm_name)
        augmenter = PromptAugmenter()

        # Step 5: Summarize the chunks to reduce the context
        print("\nStarting parallel summarization...")
        start_time = time.perf_counter()
        summarizer = Summarizer(llm,augmenter)
        summaries = await summarizer.summarize_chunks(results)
        end_time = time.perf_counter()
        print(f"\n..................Summaries....................\n{summaries}")
        print(f"[Parallel Summarization Total Time] Took {end_time - start_time:.2f} seconds")

        # Build Prompt to send context and query to LLM
        start_time = time.perf_counter()
        if summaries:
            augmenter = PromptAugmenter()
            final_prompt = augmenter.build_prompt(question, summaries)

            print("\nGenerated Prompt:\n")
            print(final_prompt)
        end_time = time.perf_counter()
        print(f"[Prompt Creation] Took {end_time - start_time:.2f} seconds")

        # Pass the prompt to LLM
        if not final_prompt:
            raise ValueError("No Final Prompt to pass to LLM")
        

        response = await llm.generate(final_prompt)
        print(f"\n🔁 [bold blue] Final Response: [/bold blue]\n [bold green] {response}[/bold green]")
        end_time = time.perf_counter()
        print(f"[Final Response From LLM] Took {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"[query_pipeline] Error: {e}")


