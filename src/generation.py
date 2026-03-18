import argparse
import json
import pickle
import os
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env.test
load_dotenv(".env.test")

# --- 1. DEFINE THE LANGGRAPH STATE ---
class GraphState(TypedDict):
    """The memory dictionary passed between our agent's nodes."""
    question: str
    generation: str
    documents: List[Document]

def run_generation(question: str, gen_config: dict, storage_config: dict):
    
    # Extract Storage Config
    chroma_path = storage_config.get("chromaDB", "./chroma_db")
    raw_store_path = storage_config.get("raw_document_store", "./raw_document_store")
    bm25_path = storage_config.get("bm25_index", "bm25_index.pkl")

    # Extract Generation Config
    multivector_k = gen_config.get("multivector_k", 4)
    bm25_k = gen_config.get("bm25_k", 4)
    ensemble_weights = gen_config.get("ensemble_weights", [0.5, 0.5])
    reranker_model_name = gen_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_threshold = gen_config.get("reranker_threshold", 0.0)
    llm_model = gen_config.get("llm_model", "llama3.2")
    llm_temperature = gen_config.get("llm_temperature", 0.2)
    embeddeing_model = gen_config.get("embedding_model", "nomic-embed-text")

    print("--- ⚙️ LOADING DATABASES ---")
    
    # 1a. Load Semantic Search (Chroma)
    embeddings = OllamaEmbeddings(model=embeddeing_model)
    vectorstore = Chroma(
        collection_name="research_summaries",
        embedding_function=embeddings,
        persist_directory=chroma_path
    )
    
    # 1b. Load Raw Document Store (LocalFileStore)
    store = LocalFileStore(raw_store_path)
    
    # 1c. Reconnect the Multi-Vector Retriever
    id_key = "doc_id"
    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": multivector_k} 
    )
    
    # 1d. Load Keyword Search (BM25)
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"Could not find '{bm25_path}'. Did you run ingestion.py?")
        
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = bm25_k 
        
    # 1e. Combine them into an Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[multi_vector_retriever, bm25_retriever],
        weights=ensemble_weights
    )
    
    print(f"--- ⚙️ LOADING RE-RANKER ({reranker_model_name}) ---")
    reranker = HuggingFaceCrossEncoder(model_name=reranker_model_name)

    # --- 2. DEFINE THE NODES ---

    def retrieve(state: GraphState):
        print("\n--- 🔍 NODE: RETRIEVE ---")
        current_question = state["question"]
        
        # Fixed: Changed final_documents to documents
        documents = ensemble_retriever.invoke(current_question)  
        print(f"Retrieved and merged top {len(documents)} chunks.")
        
        return {"documents": documents, "question": current_question}

    def grade_documents(state: GraphState):
        print(f"\n--- ⚖️ NODE: GRADE DOCUMENTS (Threshold: {reranker_threshold}) ---")
        current_question = state["question"]
        documents = state["documents"]
        
        scoring_pairs = [(current_question, doc.page_content) for doc in documents]
        scores = reranker.score(scoring_pairs)
        
        filtered_docs = []
        for i, doc in enumerate(documents):
            score = scores[i]
            print(f"  -> Chunk {i} Relevance Score: {score:.2f}")
            
            if score > reranker_threshold:
                filtered_docs.append(doc)
                
        print(f"Kept {len(filtered_docs)} highly relevant chunks for generation.")
        return {"documents": filtered_docs, "question": current_question}

    def generate(state: GraphState):
        print(f"\n--- ✍️ NODE: GENERATE ANSWER ({llm_model}) ---")
        current_question = state["question"]
        documents = state["documents"]
        
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
        
        llm = ChatOllama(model=llm_model, temperature=llm_temperature)
        prompt = ChatPromptTemplate.from_template(
            "You are an expert research assistant. Answer the question based strictly on the following context. "
            "If the context contains a table, read the rows carefully to extract the correct data.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer explicitly and accurately."
        )
        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke({"context": context, "question": current_question})
        
        return {"generation": generation}

    # --- 3. DEFINE ROUTING LOGIC ---

    def decide_to_generate(state: GraphState):
        print("\n--- 🔀 ROUTING: DECIDE NEXT STEP ---")
        if not state["documents"]:
            print("Decision: All documents failed the grading threshold. Halting.")
            return "end" 
        else:
            print("Decision: Good documents found. Proceeding to generation.")
            return "generate"

    # --- 4. BUILD AND COMPILE THE GRAPH ---

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

    app = workflow.compile()

    # --- 5. EXECUTION ---
    inputs = {"question": question}
    print(f"\nUser Question: '{inputs['question']}'")
    
    for output in app.stream(inputs):
        pass 
        
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    
    final_state = output[list(output.keys())[0]]
    print(final_state.get("generation", "I could not find relevant information in the document to answer your question. The strict grader blocked it!"))

    # Return for testing purposes
    return final_state

if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Query the ingested document database using LangGraph.")
    parser.add_argument("--question", type=str, required=True, help="The question you want to ask the document.")
    parser.add_argument("--gen_config", type=str, default="gen_config.json", help="Path to the JSON generation configuration file.")
    parser.add_argument("--storage_config", type=str, default="storage_config.json", help="Path to the JSON storage configuration file.")
    
    args = parser.parse_args()
    
    # Load Generation Config
    try:
        with open(args.gen_config, "r") as f:
            gen_config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Generation configuration file '{args.gen_config}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Generation configuration file '{args.gen_config}' is not valid JSON.")
        exit(1)
        
    # Load Storage Config
    try:
        with open(args.storage_config, "r") as f:
            storage_config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Storage configuration file '{args.storage_config}' not found. Did you run the ingestion script?")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Storage configuration file '{args.storage_config}' is not valid JSON.")
        exit(1)
        
    # Run the Generation Pipeline
    run_generation(
        question=args.question,
        gen_config=gen_config,
        storage_config=storage_config
    )