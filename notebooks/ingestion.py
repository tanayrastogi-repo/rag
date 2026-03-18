# /// script
# dependencies = [
#     "docling==2.80.0",
#     "langchain==1.2.12",
#     "langchain-chroma==1.1.0",
#     "langchain-classic==1.0.3",
#     "langchain-community==0.4.1",
#     "langchain-core==1.2.19",
#     "langchain-docling==2.0.0",
#     "langchain-ollama==1.0.1",
#     "langchain-text-splitters==1.1.1",
#     "marimo",
#     "rank-bm25==0.2.2",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    file_path = "pdfs/Population-synthesis-using-incomplete-microsample.pdf"
    return (file_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ingestion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step1. LOADING THE DOCUMENT
    """)
    return


@app.cell
def _(file_path):
    from langchain_docling import DoclingLoader
    loader = DoclingLoader(file_path=file_path, export_type="markdown")
    docs = loader.load()
    full_text = docs[0].page_content
    # mo.md(full_text)
    return DoclingLoader, full_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2. CHUNKING THE DOCUMENT
    """)
    return


@app.cell
def _(full_text):
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    # We split by Markdown headers to keep sections and tables structurally intact, plit by logical structure
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)
    print(f"Created {len(md_header_splits)} contextual chunks.")
    return (md_header_splits,)


@app.cell
def _(md_header_splits):
    # As the MarkdownHeaderTextSplitter does not enforce any chunk size limit, add RecursiveCharacterTextSplitter to enforce that. 

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    # 3. Apply the second splitter to the documents created by the first
    final_chunks = text_splitter.split_documents(md_header_splits)

    print(f"Created {len(final_chunks)} perfectly sized chunks with structural metadata!")
    return (final_chunks,)


@app.cell
def _(final_chunks):
    final_chunks[3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Page number info
    One of things, I am realizing that the chunks does not have any page number. This is because, when we used the docling we asked to extract it as a Markdown. This completely removes all the page information.

    To keep the page numbers, we have to change our strategy. Instead of converting to Markdown and using LangChain's generic splitters, we will use Docling's native structural chunker.

    Docling has a built-in tool called the `HybridChunker`. It is designed to do exactly what our previous two-step splitting did (respecting headers, tables, and size limits), but because it operates during the parsing phase, it retains the exact physical coordinates (provenance) of every word—including the page number.

    **The Production Edge Case: Vector DB Crashes**

    Because the HybridChunker tracks everything, it injects a massive, highly nested JSON dictionary into the chunk's metadata (containing bounding boxes, font sizes, etc.). ChromaDB will instantly crash if you feed it nested dictionaries or lists in the metadata. Vector databases only accept simple strings, integers, or booleans.
    """)
    return


@app.cell
def _(DoclingLoader, file_path):
    from langchain_docling.loader import ExportType
    from docling.chunking import HybridChunker

    # --- 1. PARSING AND CHUNKING NATIVELY WITH DOCLING ---

    # We change ExportType to DOC_CHUNKS to keep the metadata
    # HybridChunker automatically handles headers, tables, and chunk sizing!
    docling_loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS, 
        chunker=HybridChunker()
    )

    # These documents now have the rich, nested metadata
    raw_docs = docling_loader.load()
    print(f"✅ Created {len(raw_docs)} structural chunks.")
    return (raw_docs,)


@app.cell
def _(raw_docs):
    raw_docs[0].to_json()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to not to overwhelm the chromaDB, we flatten the metadata.
    """)
    return


@app.cell
def _(file_path, raw_docs):
    # --- 2. FLATTENING METADATA FOR CHROMADB ---

    from langchain_core.documents import Document

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
    return Document, clean_docs


@app.cell
def _(clean_docs):
    print(f"Sample Metadata of Chunk 0: {clean_docs[0].metadata}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Two-Database System

    1. **Where the Summaries go (The Vector Database):**
    	   * We convert the **summaries** into embeddings and put them in ChromaDB.
    	   * Think of the Vector Database as the **Search Index** or the card catalog in a library. It is highly optimized for searching, but it doesn't hold the actual book.

    2. **Where the Raw Chunks go (The Document Store):**
    	   * We take the **original, raw chunks** (with the perfect Markdown tables and exact phrasing) and put them in a simple key-value store (like our `InMemoryByteStore` or a persistent file store).
    	   * Think of the Document Store as the **Bookshelf**.

    When your user asks a question in your final app, this is the exact flow:
        1. The system searches the **Vector Database** and finds a matching **Summary** (because the semantic match is great).
    	2. It looks at the summary's metadata to find its ID (e.g., `doc_id: 123`).
    	3. It uses that ID to instantly grab the **Raw Chunk** from the **Document Store**.
    	4. It passes the **Raw Chunk** to the final answering LLM.

    By doing this, you get the **best of both worlds**: the highly accurate searchability of natural language summaries, *plus* the dense, detailed context of the original document for the final answer generation!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step3. GENERATING SUMMARIES WITH OLLAMA
    """)
    return


@app.cell
def _(clean_docs):
    # Using local Ollama models
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert data assistant. Summarize the following document chunk concisely. "
        "If it contains a Markdown table, explicitly describe what the table shows.\n\nChunk: {chunk}"
    )
    chain = prompt | llm | StrOutputParser()

    summaries = []
    for i, clean_doc in enumerate(clean_docs):
        print(f"  -> Summarizing chunk {i+1}/{len(clean_docs)}...")
        summary = chain.invoke({"chunk": clean_doc.page_content})
        summaries.append(summary)

    print("✅ Summaries generated successfully.")
    return (summaries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step4. EMBEDDING AND PERSISTENT STORAGE
    """)
    return


@app.cell
def _():
    # 4a. Initialize the Vector DB (Stores the text embeddings of the summaries)
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="research_summaries",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    return (vectorstore,)


@app.cell
def _():
    # 4b. Initialize the Document Store (Stores the RAW markdown/tables)
    # This creates a folder on your computer to act as our Key-Value database

    import os
    from langchain_classic.storage import LocalFileStore

    os.makedirs("./raw_document_store", exist_ok=True)
    store = LocalFileStore("./raw_document_store")
    return (store,)


@app.cell
def _(store, vectorstore):
    # 4c. Bind them together into a Multi-Vector Retriever

    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    return id_key, retriever


@app.cell
def _(clean_docs):
    # 4d. Generate unique IDs to link everything

    import uuid

    doc_ids = [str(uuid.uuid4()) for _ in clean_docs]
    return (doc_ids,)


@app.cell
def _(Document, clean_docs, doc_ids, id_key, retriever, summaries):
    # 4e. Store Summaries in Vector DB (Inheriting the clean metadata + the doc_id)
    summary_docs = [
        Document(
            page_content=summaries[i], 
            metadata={id_key: doc_ids[i], **clean_docs[i].metadata}
        )
        for i in range(len(clean_docs))
    ]
    retriever.vectorstore.add_documents(summary_docs)
    return


@app.cell
def _(clean_docs, doc_ids, retriever):
    # 4f. Store Raw Docs in LocalFileStore (via the Retriever's encoder)
    # The retriever automatically converts Document objects to bytes for the LocalFileStore!
    key_value_pairs = [
        (doc_ids[i], clean_docs[i]) 
        for i in range(len(clean_docs))
    ]
    retriever.docstore.mset(key_value_pairs)
    print("✅ Data stored in ChromaDB and LocalFileStore.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step5. BUILDING BM25 KEYWORD INDEX
    """)
    return


@app.cell
def _(clean_docs, doc_ids, id_key):
    # We inject the doc_id into the raw docs before giving them to BM25
    # This ensures that when BM25 finds a match later, we know its exact ID!

    from langchain_community.retrievers import BM25Retriever
    import pickle 

    for itr, n_doc in enumerate(clean_docs):
        n_doc.metadata[id_key] = doc_ids[itr]
    
    # Build the keyword search index based on the raw, detailed text
    bm25_retriever = BM25Retriever.from_documents(clean_docs)

    # Save the index to disk so the retrieval script can load it
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    
    print("✅ BM25 Index built and saved as 'bm25_index.pkl'.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
