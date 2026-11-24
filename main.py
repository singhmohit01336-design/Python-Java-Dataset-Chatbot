import os
import sys
import warnings
warnings.filterwarnings("ignore")

from retriever import StrictDatasetRetriever, REFUSAL_MESSAGE

# 1. Initialize Strict Dataset Retriever
# Grounded strictly in the Python & Java dataset (500 Q&A entries)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "python_java_chatbot_dataset_500.json")
retriever = StrictDatasetRetriever(dataset_path=DATASET_PATH)

# Knowledge Base Indicator
KNOWLEDGE_BASE_LABEL = "Knowledge Base: Python & Java (500 entries)"


def query_rag(question: str) -> str:
    """Queries the strict Python & Java dataset-grounded knowledge base.
    
    Returns the exact grounded answer if the question matches an entry
    in the dataset above the confidence threshold, otherwise returns
    the strict refusal message. No general LLM fallback is used.
    """
    print("\n" + "=" * 60)
    print(f"[{KNOWLEDGE_BASE_LABEL}]")
    print(f"Query: {question}")
    print("=" * 60)

    # Deterministic local retrieval
    entry, score, reason = retriever.retrieve(question, threshold=0.40)

    if entry:
        print(f"\n[Retrieval Result]")
        print(f" Matched ID    : {entry['id']}")
        print(f" Language      : {entry['language']}")
        print(f" Confidence    : {score:.3f} (Passed threshold 0.40)")
        print(f" Reference Q   : {entry['question']}")
        response = entry["answer"]
    else:
        print(f"\n[Retrieval Result]")
        print(f" Status        : Refused ({reason})")
        print(f" Confidence    : {score:.3f} (Threshold: 0.40)")
        response = REFUSAL_MESSAGE

    print("\nAnswer:")
    print(response)
    print("=" * 60)
    return response


def run_interactive_cli():
    """Runs an interactive CLI chat loop."""
    print("\n" + "=" * 60)
    print(f"Welcome to the AI Chatbot [{KNOWLEDGE_BASE_LABEL}]")
    print("Strictly answers questions covered by Python & Java dataset.")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chatbot. Goodbye!")
                break
            query_rag(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot. Goodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--interactive", "-i"):
            run_interactive_cli()
        else:
            cli_question = " ".join(sys.argv[1:])
            query_rag(cli_question)
    else:
        # Default verification demonstrations
        print("\n--- Running Baseline Demonstrations ---")
        query_rag("What is Python?")
        query_rag("Explain inheritance in Java.")
        query_rag("Who is the Prime Minister of India?")
