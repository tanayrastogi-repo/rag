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

### Reasoning behind certain decisions: 

* **Multi-vector**: We use a Multi-Vector Retriever because research paper contains test, tables and images/figures. Standard RAG will just embedded them directly. Here we use a LLM to write concise, searchable summary of every table and figure and put in the vector dataset. We also link them via a common ID to a separate DocumentStore that holds the raw, original table or image. When the user asks a question, the system searches the summaries, finds a match, but retrieves the raw data to pass to the final generation LLM.

* **Docling for Parsing**: While parsing manuscripts, it is important to preserve the document structure. Docling, compared to LlamaParse/Unstructured, is native integration with Langchain. It can returns markdown from the original formatted headings. However, docling is very heavy and computationally slow. Don't use it if you need real-time document processing for user uploads.

* **Using HybridChunker**: We use the Docling's HybridChunker because it also save the "page number" as metadata during the chunking. Compared to "MarkdownHeaderTextSplitter" which only returns a markdown and looses all the text page structure information. 

* **Issue with HybridChunker**: Because the HybridChunker tracks everything, it injects a massive, highly nested JSON dictionary into the chunk's metadata (containing bounding boxes, font sizes, etc.). ChromaDB will instantly crash if you feed it nested dictionaries or lists in the metadata. We must write a script to "flatten" this metadata, extracting only the page numbers and headers into clean strings.

* **Chunk size using MarkdownHeaderTextSplitter**: As the MarkdownHeaderTextSplitter only chunks the text based headers, there is no standard limit on the size of the chunk. It aims to keep the text under the header as it is. If we want to have chunk of constant size, then we have to have a hybrid solution using MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter. We use the MarkdownHeaderTextSplitter to get the structure and metadata, and then we pass those results through a RecursiveCharacterTextSplitter to enforce a strict size limit.

* **Importance of chunk size**: A 15-page chunk will crash your embedding model (which usually has a strict token limit, like 8192 tokens for nomic-embed-text).

* **EnsembleRetriever**: During the generation time, we use the Langchain's EnsembleRetriever to retrive information from both the MultiVectorRetriever and BM25Retriever. The results are then extracted from both the retriever and then are merged using Reciprocal Rank Fusion (RRF) method. 

* **Re-Ranking**: This is the curcial part of the self-reflective RAG. The pipeline grades the extracted documents to make sure that the information in them are relevant. Either we can use a "LLM-as-judge" or use a re-ranker model. Re-ranker are faster than LLM-as-judge and are also not prone to hallucinating. Using Re-rankers use get - 1. Speed --> Generating tokens with an LLM takes seconds. A local re-ranker scoring 4 documents takes milliseconds. 2. Cost --> If you were using a cloud API, you just removed an entire LLM call from your workflow. 3. Accuracy --> The RRF algorithm we used in the retrieval step is great for rough sorting, but the re-ranker acts as the ultimate, highly-intelligent filter. It prevents the generation LLM from ever seeing garbage data.

* **HNSW**: The HNSW (Hierarchical Navigable Small World) algorithm is operating entirely inside of ChromaDB. It is the invisible, lightning-fast math engine that makes your semantic search possible. Let's break down exactly what it is doing and where it fires off in the pipeline we just built. HNSW is an Approximate Nearest Neighbor (ANN) algorithm. Instead of checking everything, it builds a multi-layered graph. HNSW: It trades a tiny bit of accuracy for a massive amount of speed.
  

* 

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