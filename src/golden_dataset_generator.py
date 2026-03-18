import argparse
from dotenv import load_dotenv

# Langchain imports for loading data and LLM agents
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# RAGAS imports for generation
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv(".env.test")

def create_golden_dataset(pdf_path: str, output_file: str, test_size: int = 10) -> None:
    print(f"\n--- 📄 LOADING DOCUMENT: {pdf_path} ---")

    # Use Docling to load the document for the generator
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker()
    )

    loader = DoclingLoader(file_path=pdf_path)
    documents = loader.load()

    # 2. CONFIGURE GENERATOR MODELS (Ollama)
    # Ragas needs a wrapper to talk to LangChain models
    generator_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b", temperature=0.3))
    # generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1))
    generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    # 3. INITIALIZE RAGAS GENERATOR
    print("\n--- 🧠 GENERATING SYNTHETIC TESTSET (This may take a while...) ---")
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings
    )

    # 4. GENERATE
    # testset_size is the number of Q&A pairs to generate
    # generates questions based on three types: simple (direct), reasoning (requires inference), and multi_context (requires synthesizing multiple parts of the document)
    testset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=test_size,
        # We can specify the distribution of question types
        query_distribution=[
            (0.5, "simple"), 
            (0.25, "reasoning"), 
            (0.25, "multi_context")
        ]
    )

    # 5. SAVE TO JSON
    # Convert to pandas first for easy export
    df = testset.to_pandas()
    
    # We rename columns to match what our test_harness.py expects
    # Ragas uses: question, reference (ground truth), and reference_contexts
    df = df.rename(columns={
        "user_input": "question", 
        "reference": "ground_truth"
    })
    
    # Save as JSON records
    df[["question", "ground_truth"]].to_json(output_file, orient="records", indent=4)
    print(f"✅ Golden Dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic Golden Dataset for RAG evaluation.")
    
    # Define Arguments
    parser.add_argument("--pdf", type=str, required=True, help="Path to the source PDF file.")
    parser.add_argument("--size", type=int, default=10, help="Number of Q&A pairs to generate (default: 10).")
    parser.add_argument("--output", type=str, default="golden_dataset.json", help="Path to save the generated JSON file (default: golden_dataset.json).")

    args = parser.parse_args()

    # Run the generation
    create_golden_dataset(
        pdf_path=args.pdf, 
        output_file=args.output, 
        test_size=int(args.size)
    )