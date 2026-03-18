import uuid
import pickle
import os
import argparse
import json

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.retrievers import BM25Retriever

def run_ingestion(
    file_path: str, 
    llm_model: str, 
    embedding_model: str, 
    chroma_path: str, 
    raw_store_path: str, 
    bm25_path: str
):
    """
    Executes the complete production-ready ingestion pipeline:
    1. Native Docling parsing & chunking (preserves page numbers/tables)
    2. Metadata sanitization for ChromaDB
    3. Multi-Vector summarization via Ollama
    4. Persistent storage (Chroma for summaries, LocalFileStore for raw text)
    5. BM25 index generation for Hybrid Search
    """
    print("\n--- 1. PARSING AND CHUNKING WITH DOCLING ---")
    print(f"Processing: {file_path}")
    
    # ExportType.DOC_CHUNKS + HybridChunker ensures we keep the exact physical layout
    # and table structures while adhering to optimal size limits.
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker()
    )
    raw_docs = loader.load()
    print(f"✅ Created {len(raw_docs)} structural chunks.")

    print("\n--- 2. FLATTENING METADATA FOR VECTOR DB ---")
    clean_docs = []
    
    for doc in raw_docs:
        dl_meta = doc.metadata.get("dl_meta", {})
        pages = set()
        
        # Extract page numbers from the complex Docling dictionary
        for item in dl_meta.get("doc_items", []):
            for prov in item.get("prov", []):
                if "page_no" in prov:
                    pages.add(prov["page_no"])
                    
        headings = dl_meta.get("headings", [])
        
        # Create safe, flat metadata (ChromaDB strictly requires strings/ints/floats)
        clean_meta = {
            "source": file_path,
            "page_numbers": str(sorted(list(pages))), 
            "headings": str(headings)
        }
        
        # Create a new Document object with the raw content and safe metadata
        clean_docs.append(Document(page_content=doc.page_content, metadata=clean_meta))
        
    print("✅ Metadata successfully flattened and sanitized.")

    print(f"\n--- 3. GENERATING SUMMARIES WITH OLLAMA ({llm_model}) ---")
    llm = ChatOllama(model=llm_model, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert data assistant. Summarize the following document chunk concisely. "
        "If it contains a Markdown table, explicitly describe what the table shows.\n\nChunk: {chunk}"
    )
    chain = prompt | llm | StrOutputParser()
    
    summaries = []
    for i, doc in enumerate(clean_docs):
        print(f"  -> Summarizing chunk {i+1}/{len(clean_docs)}...")
        summary = chain.invoke({"chunk": doc.page_content})
        summaries.append(summary)
        
    print("✅ Summaries generated successfully.")

    print(f"\n--- 4. EMBEDDING AND PERSISTENT STORAGE ({embedding_model}) ---")
    
    # 4a. Initialize the Vector DB (Stores the text embeddings of the summaries)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name="research_summaries",
        embedding_function=embeddings,
        persist_directory=chroma_path
    )
    
    # 4b. Initialize the Document Store (Stores the RAW markdown/tables)
    # This creates a folder on your computer to act as our Key-Value database
    os.makedirs(raw_store_path, exist_ok=True)
    store = LocalFileStore(raw_store_path)
    
    # 4c. Bind them together into a Multi-Vector Retriever
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    # 4d. Generate unique IDs to link everything
    doc_ids = [str(uuid.uuid4()) for _ in clean_docs]
    
    # 4e. Store Summaries in Vector DB (Inheriting the clean metadata + the doc_id)
    summary_docs = [
        Document(
            page_content=summaries[i], 
            metadata={id_key: doc_ids[i], **clean_docs[i].metadata}
        )
        for i in range(len(clean_docs))
    ]
    retriever.vectorstore.add_documents(summary_docs)
    
    # 4f. Store Raw Docs in LocalFileStore (via the Retriever's encoder)
    # The retriever automatically converts Document objects to bytes for the LocalFileStore!
    key_value_pairs = [
        (doc_ids[i], clean_docs[i]) 
        for i in range(len(clean_docs))
    ]
    retriever.docstore.mset(key_value_pairs)
    print("✅ Data stored in ChromaDB and LocalFileStore.")

    print("\n--- 5. BUILDING BM25 KEYWORD INDEX ---")
    # We inject the doc_id into the raw docs before giving them to BM25
    # This ensures that when BM25 finds a match later, we know its exact ID!
    for i, doc in enumerate(clean_docs):
        doc.metadata[id_key] = doc_ids[i]
        
    # Build the keyword search index based on the raw, detailed text
    bm25_retriever = BM25Retriever.from_documents(clean_docs)
    
    # Save the index to disk so the retrieval script can load it
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print("✅ BM25 Index built and saved as 'bm25_index.pkl'.")
    print("\n🎉🎉 INGESTION COMPLETE! Your database is ready for queries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into a multi-vector and BM25 database.")
    parser.add_argument("--ingest_file", type=str, required=True, help="Path to the PDF file to ingest.")
    parser.add_argument("--config_file", type=str, default="ingest_config.json", help="Path to the JSON model configuration file.")
    parser.add_argument("--storage_config", type=str, default="storage_config.json", help="Path to the JSON storage configuration file.")
    
    args = parser.parse_args()
    
    # Load model configuration
    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Model configuration file '{args.config_file}' not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Model configuration file '{args.config_file}' is not valid JSON.")
        exit(1)
        
    # Load storage configuration
    try:
        with open(args.storage_config, "r") as f:
            storage_config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Storage configuration file '{args.storage_config}' not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Storage configuration file '{args.storage_config}' is not valid JSON.")
        exit(1)

    # Extract configuration variables with fallbacks
    llm_model = config.get("llm_model", "llama3.2:3b")
    embedding_model = config.get("embedding_model", "nomic-embed-text")
    
    chroma_path = storage_config.get("chromaDB", "./chroma_db")
    raw_store_path = storage_config.get("raw_document_store", "./raw_document_store")
    bm25_path = storage_config.get("bm25_index", "bm25_index.pkl")
    
    # Run the pipeline
    if os.path.exists(args.ingest_file):
        run_ingestion(
            file_path=args.ingest_file, 
            llm_model=llm_model, 
            embedding_model=embedding_model,
            chroma_path=chroma_path,
            raw_store_path=raw_store_path,
            bm25_path=bm25_path
        )
    else:
        print(f"❌ Error: Could not find '{args.ingest_file}'. Please check the file path.")