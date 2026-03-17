# Self-Reflective Multi-Vector RAG 🧠📊

## 📖 Short Description
A production-ready, 100% locally hosted Retrieval-Augmented Generation (RAG) pipeline designed to process complex research papers. Moving beyond naive text chunking, this project uses an advanced multi-vector architecture to accurately retrieve information from complex tables, nested layouts, and dense scientific text. It features an intelligent LangGraph agent that grades its own retrieved context using a cross-encoder before generating an answer, ensuring high accuracy and preventing hallucinations.

---

## ✨ Key Features

* **Intelligent Document Parsing (Docling):** Uses Docling's `HybridChunker` to natively parse PDFs into clean Markdown, preserving complex table structures and exact page-number provenance.
* **Multi-Vector Architecture:** Uses a local LLM to generate searchable semantic summaries of document chunks (stored in ChromaDB), while preserving the raw, unedited Markdown tables for the final generation step (stored in a LocalFileStore).
* **Hybrid Search (Ensemble Retriever):** Combines dense semantic search (HNSW algorithm via Chroma) with sparse exact-keyword matching (BM25) using Reciprocal Rank Fusion (RRF) for unparalleled retrieval accuracy.
* **Cross-Encoder Re-ranking:** Replaces the traditional "LLM-as-a-judge" with a lightning-fast HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to strictly grade and filter retrieved documents.
* **Self-Reflective Agent (LangGraph):** Implements a state-machine workflow that halts generation if the retrieved context fails the relevance threshold, preventing hallucinated answers.
* **Privacy-First & Zero Cost:** Fully local execution powered by Ollama (`llama3.2:3b` for generation/summaries, `nomic-embed-text` for embeddings).
* **Enterprise Observability:** Fully instrumented with LangSmith for real-time tracing of the graph execution, retrieval metrics, and LLM token usage.

---

## 🗺️ System Architecture

The pipeline is split into two distinct phases: asynchronous **Ingestion** and real-time **Retrieval & Generation**.

```mermaid
flowchart TD
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:1px;
    classDef database fill:#fce4ec,stroke:#c2185b,stroke-width:1px;
    classDef agent fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    subgraph Ingestion ["Phase 1: Ingestion (ingest.py)"]
        direction TB
        PDF([Complex PDF]) --> Parse[Docling HybridChunker\nExtracts Markdown & Pages]:::process
        Parse --> Clean[Flatten Metadata]:::process
        Clean --> Sum[Ollama 3.2\nGenerates Table Summaries]:::process
        
        Sum -->|Embeds Summaries| DB_Vec[(ChromaDB\nSemantic Index)]:::database
        Clean -->|Saves Raw Tables/Text| DB_Raw[(LocalFileStore\nKey-Value)]:::database
        Clean -->|Builds Keyword Index| DB_Sparse[(BM25 Index\nSparse Vectors)]:::database
    end

    subgraph Retrieval ["Phase 2: Retrieval & Generation (retrieve_generate.py)"]
        direction TB
        Query([User Question]) --> ER{Ensemble Retriever}:::process
        
        ER -->|Semantic Search| DB_Vec
        ER -->|Keyword Search| DB_Sparse
        
        DB_Vec -->|Matches IDs| DB_Raw
        DB_Raw --> RRF[Reciprocal Rank Fusion\nMerges & Deduplicates]:::process
        DB_Sparse --> RRF
        
        RRF --> ReRank[MiniLM Cross-Encoder\nStrictly Grades Relevance]:::agent
        ReRank --> Gate{Relevant Docs\nFound?}:::agent
        
        Gate -->|Yes| Gen[Llama 3.2\nGenerates Final Answer]:::process
        Gate -->|No| Halt([Halt & Prevent Hallucination]):::process
        Gen --> Out([Final Contextual Answer]):::process
    end