import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from dotenv import load_dotenv
    import os
    import sys

    load_dotenv(".env.test")
    return (os,)


@app.cell
def _():
    import json

    # 1. LOAD CONFIGS
    print("--- ⚙️ LOADING CONFIGURATIONS ---")

    gen_cfg_path = "src/gen_config.json"
    storage_cfg_path = "src/storage_config.json"

    try:
        with open(gen_cfg_path, "r") as f:
            gen_config = json.load(f)
        with open(storage_cfg_path, "r") as f:
            storage_config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config files: {e}")
        # Don't exit(1) in a notebook as it kills the kernel
        gen_config, storage_config = {}, {}
    return gen_config, json, storage_config


@app.cell
def _(json, os):
    # 2. LOAD THE GENERATED GOLDEN DATASET
    print("--- 📂 LOADING GOLDEN DATASET FROM FILE ---")
    if not os.path.exists("golden_dataset.json"):
        print("❌ Error: golden_dataset.json not found. Run generate_testset.py first!")
        exit(1)

    with open("golden_dataset.json", "r") as file:
        qa_pairs = json.load(file)

    print(f"Loaded {len(qa_pairs)} test cases.")
    return (qa_pairs,)


@app.cell
def _(gen_config):
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    # 3. CONFIGURE RAGAS
    print(f"--- 🤖 CONFIGURING RAGAS JUDGE ({gen_config['llm_model']}) ---")
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=gen_config["llm_model"], temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=gen_config["embeddeing_model"]))

    # Defining the LLM model for LLM-as-judge
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings

    metrics_to_use = [faithfulness, answer_relevancy]
    return (metrics_to_use,)


@app.cell
def _(gen_config, storage_config):
    from generation import run_generation

    # 4. PREDICTION WRAPPER FOR LANGSMITH
    def predict_rag_answer(inputs: dict) -> dict:
        """Passes the question to your generation.py script and formats the output."""
        question = inputs["question"]

        # Call your modularized generation pipeline
        final_state = run_generation(question, gen_config, storage_config)

        # Extract the data Ragas needs to grade the answer
        generation = final_state.get("generation", "No answer could be generated.")
        documents = final_state.get("documents", [])
        contexts = [doc.page_content for doc in documents]

        return {
            "answer": generation,
            "contexts": contexts
        }

    return (predict_rag_answer,)


@app.cell
def _(metrics_to_use):
    from ragas import evaluate as ragas_evaluate
    from datasets import Dataset

    # 5. RAGAS EVALUATION WRAPPER
    def ragas_evaluator(run, example) -> dict:
        """Passes the generated answer and context to Ragas for scoring."""
        data = {
            "question": [example.inputs["question"]],
            "answer": [run.outputs["answer"]],
            "contexts": [run.outputs["contexts"]],
            "ground_truth": [example.outputs["ground_truth"]]
        }
    
        dataset = Dataset.from_dict(data)
        score = ragas_evaluate(dataset, metrics=metrics_to_use, raise_exceptions=False)
    
        results = []
        if "faithfulness" in score:
            results.append({"key": "faithfulness", "score": score["faithfulness"]})
        if "answer_relevancy" in score:
            results.append({"key": "answer_relevancy", "score": score["answer_relevancy"]})
        
        return {"results": results}

    return (ragas_evaluator,)


@app.cell
def _(client, qa_pairs):
    print("\n--- 🧪 CREATING DATASET IN LANGSMITH ---")
    dataset_name = "Modular_RAG_Golden_Dataset_v1"

    try:
        dataset = client.create_dataset(dataset_name=dataset_name, description="Test dataset for Multi-vector RAG.")
        for pair in qa_pairs:
            client.create_example(
                inputs={"question": pair["question"]},
                outputs={"ground_truth": pair["ground_truth"]},
                dataset_id=dataset.id,
            )
        print("✅ Dataset uploaded successfully!")
    except Exception as e:
        print("Dataset already exists. Proceeding to evaluation...")
    return (dataset_name,)


@app.cell
def _(dataset_name, evaluate, gen_config, predict_rag_answer, ragas_evaluator):
    print("\n--- 🚀 STARTING EVALUATION LOOP ---")
    
    # Create a dynamic custom name using the ensemble weights from your config
    weights = gen_config.get("ensemble_weights", [0.5, 0.5])
    custom_run_name = f"Test-Weights-{weights[0]}_{weights[1]}-"

    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[ragas_evaluator],
        experiment_prefix=custom_run_name, 
        metadata=gen_config # <-- MAGIC: Logs your entire gen_config.json to LangSmith!
    )

    print("\n✅ Evaluation complete! Open LangSmith to view your dashboard.")
    return


if __name__ == "__main__":
    app.run()
