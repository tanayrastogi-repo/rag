# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ipython==9.10.0",
#     "langchain==1.2.12",
#     "langchain-chroma==1.1.0",
#     "langchain-classic==1.0.3",
#     "langchain-community==0.4.1",
#     "langchain-core==1.2.19",
#     "langchain-ollama==1.0.1",
#     "langgraph==1.1.2",
#     "sentence-transformers==5.3.0",
#     "typing-extensions==4.15.0",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Retrieval & Generation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview of the Retrieval & Generation Script

    Here is exactly how this script will work:
    1. **The Load Phase:** We point our script to the folders and files we created during ingestion.
    2. **The Ensemble Retriever:** We combine our `MultiVectorRetriever` (which handles the semantic search on summaries) and our `BM25Retriever` (which handles exact keyword matching) into a single, powerful `EnsembleRetriever`.
    3. **The LangGraph Agent:** We define our nodes (`retrieve`, `grade_documents`, `generate`) and our conditional logic to ensure the LLM only answers if it actually found good data.

    References:
    * [Langchain BLog](https://blog.langchain.com/agentic-rag-with-langgraph/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step1. LOAD DATABASES & SETUP RETRIEVERS
    """)
    return


@app.cell
def _():
    n_multivector_documents = 3 # Limit to top n semantic matches
    n_bm25_documents = 3 # Limit to top n keyword matches
    return n_bm25_documents, n_multivector_documents


@app.cell
def _(n_bm25_documents, n_multivector_documents):
    # 1a. Load Semantic Search (Chroma)

    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="research_summaries",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # 1b. Load Raw Document Store (LocalFileStore)
    from langchain_classic.storage import LocalFileStore
    store = LocalFileStore("./raw_document_store")
    
    # 1c. Reconnect the Multi-Vector Retriever
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    id_key = "doc_id"
    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": n_multivector_documents}  # <--- NEW: Limit to top 3 semantic matches
    )
    
    # 1d. Load Keyword Search (BM25)
    import pickle
    with open("bm25_index.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = n_bm25_documents # <--- NEW: Limit to top 3 keyword matches
    return bm25_retriever, multi_vector_retriever


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step2. Ensemble Retriever
    """)
    return


@app.cell
def _(bm25_retriever, multi_vector_retriever):
    # Combine the semantic and keyword search retrivers
    # (Weights: 50% semantic summaries, 50% exact keyword match)
    from langchain_classic.retrievers import EnsembleRetriever

    ensemble_retriever = EnsembleRetriever(
        retrievers=[multi_vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return (ensemble_retriever,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step3. Define Nodes for LangGraph
    """)
    return


@app.cell
def _():
    ## State definition
    from typing import List
    from typing_extensions import TypedDict
    from langchain_core.documents import Document

    class GraphState(TypedDict):
        """The dictionary that gets passed between our LangGraph nodes."""
        question: str
        generation: str
        documents: List[Document]

    return (GraphState,)


@app.cell
def _(GraphState, ensemble_retriever):
    ## Retrieve node

    def retrieve(state: GraphState):
        print("--- 🔍 RETRIEVE ---")
        question = state["question"]
    
        # Use our combined retriever to get the best documents
        documents = ensemble_retriever.invoke(question)
        print(f"Retrieved {len(documents)} document chunks.")
    
        return {"documents": documents, "question": question}

    return (retrieve,)


@app.cell
def _(GraphState):
    ## Grading node
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    # Load a re-ranker model
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Threshold for the score
    THRESHOLD = 0.0

    def grade_documents(state: GraphState):
        print("--- ⚖️ GRADE DOCUMENTS (VIA RE-RANKER) ---")
        question = state["question"]
        documents = state["documents"]
    
        # 1. Prepare the pairs of (Question, Document Text) for the model
        scoring_pairs = [(question, doc.page_content) for doc in documents]
    
        # 2. Get the exact relevance scores from the Cross-Encoder
        scores = reranker.score(scoring_pairs)
    
        # 3. Filter documents based on a threshold (e.g., score > 0.3)
        # Note: BGE-reranker outputs logits. A score > 0 is usually a solid semantic match, but you can tune this threshold based on your specific documents.
    
        filtered_docs = []
        for i, doc in enumerate(documents):
            score = scores[i]
            print(f"Document {i} Score: {score:.2f}")
        
            if score > THRESHOLD:
                filtered_docs.append(doc)
            
        print(f"Kept {len(filtered_docs)} highly relevant chunks after re-ranking.")
        return {"documents": filtered_docs, "question": question}

    return grade_documents, reranker


@app.cell
def _(GraphState):
    ## Answer Generating node

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    def generate(state: GraphState):
        print("--- ✍️ GENERATE ANSWER ---")
        question = state["question"]
        documents = state["documents"]
    
        # Combine the rich, raw text/tables from the relevant documents
        context = "\n\n".join([doc.page_content for doc in documents])
    
        llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
        prompt = ChatPromptTemplate.from_template(
            "You are an expert research assistant. Answer the question based strictly on the following context. "
            "If the context contains a table, read the rows carefully to extract the correct data.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer explicitly and cite the page numbers if available in the context."
        )
        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke({"context": context, "question": question})
    
        return {"generation": generation}

    return (generate,)


@app.cell
def _(GraphState):
    # Router Logic
    def decide_to_generate(state: GraphState):
        print("--- 🔀 DECIDE NEXT STEP ---")
        if not state["documents"]:
            print("Decision: All documents were irrelevant. Halting to prevent hallucination.")
            return "end" 
        else:
            print("Decision: Relevant documents found. Proceeding to generation.")
            return "generate"

    return (decide_to_generate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step4. Build and Compile the Graph
    """)
    return


@app.cell
def _(GraphState, decide_to_generate, generate, grade_documents, retrieve):
    from langgraph.graph import StateGraph, START, END

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "end": END,
        }
    )
    workflow.add_edge("generate", END)

    # Compile the agent!
    app = workflow.compile()
    return (app,)


@app.cell
def _(app, mo):
    mo.mermaid(app.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step5. INFERENCE
    """)
    return


@app.cell
def _():
    user_question = "Which is the captial of Berlin?"
    return (user_question,)


@app.cell
def _(user_question):
    inputs = {"question": user_question}
    print(f"\nUser Question: {inputs['question']}\n")
    return (inputs,)


@app.cell
def _(app, inputs):
    # Stream the steps so we can watch it think
    for output in app.stream(inputs):
        pass 
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")

    # Safely get the final generation, or a default message if it was halted
    final_state = output[list(output.keys())[0]]
    print(final_state.get("generation", "I could not find relevant information in the document to answer your question."))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **TESTING**
    Simple test to check our pipeline
    """)
    return


@app.cell
def _():
    question = "Which generative model is used in this research?"
    return (question,)


@app.cell
def _(ensemble_retriever, question):
    # Ensemble retriver
    documents = ensemble_retriever.invoke(question)
    print(f"Retrieved {len(documents)} document chunks.")
    return (documents,)


@app.cell
def _(documents, mo):
    mo.json(documents[0].model_dump_json())
    return


@app.cell
def _(documents, question, reranker):
    # Reranker
    scoring_pairs = [(question, doc.page_content) for doc in documents]
    scores = reranker.score(scoring_pairs)
    return (scores,)


@app.cell
def _(documents, scores):
    # Threshold filering
    filtered_docs = []
    for i, doc in enumerate(documents):
        score = scores[i]
        print(f"Document {i} Score: {score:.2f}")
    
        if score > 0.0:
            filtered_docs.append(doc)
    print(f"Kept {len(filtered_docs)} highly relevant chunks after re-ranking.")
    return (filtered_docs,)


@app.cell
def _(filtered_docs):
    filtered_docs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## INNER WORKING OF THE ENSEMBLE RETRIEVER

    Because we are using an **Ensemble Retriever** combined with a **Multi-Vector Retriever**, the process is actually split into two parallel tracks before merging back together.

    ### Phase 1: The Split
    1. **Input:** The LangGraph node passes your raw string (e.g., *"What is the accuracy of Llama3.2?"*) to the `EnsembleRetriever`.
    2. **Duplication:** The Ensemble Retriever instantly duplicates your question and sends it down two separate tracks simultaneously: The **Semantic Track** (Chroma) and the **Keyword Track** (BM25).

    ---

    ### Phase 2: Track A - The Semantic Search (Multi-Vector Retriever)
    This track looks for the *meaning* of your question.

    3. **Embedding Generation:** LangChain sends your exact question string to Ollama (`nomic-embed-text`). Ollama mathematically translates your question into a dense vector (an array of thousands of numbers).
    4. **Vector Database Search:** This question vector is sent to ChromaDB. Chroma compares the math of your question vector against the math of all the **summary vectors** we created during ingestion. It finds the nearest matches (e.g., the top 4 most semantically similar summaries).
    5. **ID Extraction:** Chroma returns the metadata of those top summaries. LangChain looks specifically at the `doc_id` attached to them (e.g., `id: 123` and `id: 456`).
    6. **Raw Document Fetch:** Here is where the Multi-Vector magic happens! The retriever takes those IDs, goes to your `LocalFileStore` (the folder on your hard drive), and pulls the **raw, unedited Markdown chunks** that correspond to those IDs.
       * *Track A Output:* A list of raw Document objects ranked by semantic meaning.

    ---

    ### Phase 3: Track B - The Keyword Search (BM25 Retriever)
    This track looks for *exact vocabulary matches*. **Notice that embeddings are NOT used here!**

    7. **Tokenization:** LangChain takes your raw string and breaks it down into root words (tokens). It strips out filler words (like "what", "is", "the") and isolates the key terms: `["accuracy", "llama3.2"]`.
    8. **Index Lookup:** It checks these terms against the `bm25_index.pkl` file we loaded into memory. BM25 is essentially a massive, highly optimized dictionary.
    9. **Frequency Scoring:** It scores the raw documents based on how many times those exact words appear, factoring in how rare the word is across the whole database (e.g., "llama3.2" is rare and gets a high score; "accuracy" is common and gets a lower score).
       * *Track B Output:* A list of raw Document objects ranked by exact keyword matches.

    ---

    ### Phase 4: The Merge (Reciprocal Rank Fusion)
    Now, the `EnsembleRetriever` has two different lists of top documents, and it needs to figure out which ones are actually the best.

    10. **Reciprocal Rank Fusion (RRF):** LangChain applies a mathematical algorithm called RRF. It doesn't look at the raw scores from Chroma or BM25 (because those scores are calculated completely differently and can't be compared directly). Instead, it looks at their **Rank**.
        * If a document was Ranked #1 in Semantic Search *and* Ranked #1 in Keyword Search, its RRF score skyrockets to the top.
        * If a document was Ranked #2 in Semantic but wasn't found in Keyword, it gets a moderate score.
    11. **Applying Weights:** Remember when we set `weights=[0.5, 0.5]` in the code? The RRF algorithm multiplies the ranks by these weights. You could change this to `[0.3, 0.7]` if you wanted exact keywords to matter more than semantic meaning!
    12. **Deduplication and Final Output:** The algorithm removes any duplicate documents, sorts the final list by the combined RRF score, and returns the top results.

    ### The Final Result
    13. **Return:** The `EnsembleRetriever` finally hands back a clean `List[Document]` to your LangGraph `retrieve` node. These documents are the absolute best mathematical blend of contextual meaning and exact keyword hits, complete with their raw tables and page number metadata!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid('''
    flowchart TD
        %% Styling
        classDef input fill:#f9f,stroke:#333,stroke-width:2px;
        classDef process fill:#bbf,stroke:#333,stroke-width:1px;
        classDef database fill:#fbb,stroke:#333,stroke-width:1px;
        classDef final fill:#bfb,stroke:#333,stroke-width:2px;

        %% Phase 1: Input & Split
        Q(["User Question: 'What is the accuracy?'"]):::input --> ER{"Ensemble Retriever"}:::process
    
        ER -->|Track A: Meaning| Embed[Ollama Embeddings\n'nomic-embed-text']:::process
        ER -->|Track B: Keywords| Token[Text Tokenization]:::process

        %% Phase 2: Track A (Semantic / Multi-Vector)
        subgraph Track_A ["Phase 2: Semantic Search (Multi-Vector)"]
            direction TB
            Embed -->|Generates| QVec[Dense Question Vector]:::process
            QVec -->|Similarity Search| Chroma[(ChromaDB\nSummaries)]:::database
            Chroma -->|Returns matches| Sums[Top Document Summaries]:::process
            Sums -->|Extracts Metadata| IDs[Extract 'doc_id']:::process
            IDs -->|Key-Value Lookup| LFS[(LocalFileStore\nRaw Text/Tables)]:::database
            LFS -->|Fetches Raw Data| DocsA[List of Semantic Docs]:::process
        end

        %% Phase 3: Track B (Keyword / BM25)
        subgraph Track_B ["Phase 3: Keyword Search (BM25)"]
            direction TB
            Token -->|Extracts| Keys[Keywords: 'accuracy']:::process
            Keys -->|Frequency Search| BM25[(BM25 Index\nSparse Vectors)]:::database
            BM25 -->|Scores matches| DocsB[List of Keyword Docs]:::process
        end

        %% Phase 4: Merge (Reciprocal Rank Fusion)
        DocsA --> RRF{"Reciprocal Rank Fusion (RRF)"}:::process
        DocsB --> RRF
    
        subgraph Phase_4 ["Phase 4: The Merge"]
            RRF -->|Applies Weights: 0.5 / 0.5| Dedup[Remove Duplicates\n& Sort by new score]:::process
        end

        Dedup --> Final(["Final Top Documents returned to LangGraph"]):::final
    ''')
    return


if __name__ == "__main__":
    app.run()
