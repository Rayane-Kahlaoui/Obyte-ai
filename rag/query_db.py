import os
import sys
import argparse
# Ensure the project root directory is on sys.path so `rag` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.config import OPENROUTER_API_KEY
# Ensure OpenRouter API key is available for LLMClient
os.environ.setdefault('OPENROUTER_API_KEY', OPENROUTER_API_KEY)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to enable local imports
# Path already configured above (no extra append needed)

from rag.orchestrator import RAGOrchestrator
from rag.llm_client import LLMClient
from rag.config import LLM_API_TYPE

def main():
    # Map configuration API type back to CLI mode option
    reverse_api_map = {
        "hf_inference": "hf",
        "local_api": "local",
        "mock": "mock",
        "openrouter": "openrouter"
    }
    default_mode = reverse_api_map.get(LLM_API_TYPE, "local")

    parser = argparse.ArgumentParser(description="Query the Multi-Agent Legal RAG System")
    parser.add_argument("query", type=str, nargs="?", default="", help="The question to ask")
    parser.add_argument("--user", type=str, default="charlie", help="Username to query as (alice/bob/charlie)")
    parser.add_argument("--live", action="store_true", help="Toggle live local LLM server instead of mock")
    parser.add_argument("--mode", type=str, choices=["hf", "local", "mock", "openrouter"], default=default_mode, help="LLM backend mode (hf: Hugging Face serverless free API, local: Ollama/local server, mock: rule-based generator, openrouter: OpenRouter API)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    args = parser.parse_args()

    if not args.query:
        print("Please provide a query. Example: python rag/query_db.py \"What is compliance policy?\" --user alice")
        return

    # Map CLI mode argument to llm_client api_type
    api_map = {
        "hf": "hf_inference",
        "local": "local_api",
        "mock": "mock",
        "openrouter": "openrouter"
    }
    # Determine LLM client API type based on CLI mode
    mode = api_map[args.mode]
    if args.live:
        # If legacy --live flag is used, force local API
        mode = "local_api"
    llm_client = LLMClient(api_type=mode)
    orchestrator = RAGOrchestrator(llm_client=llm_client)

    print(f"Processing query as user: {args.user}...")
    result = orchestrator.process_query(args.user, args.query, top_k=args.top_k)

    print("\n" + "="*80)
    print(f"USER QUERY: {result['query']}")
    print(f"USER: {result['username']} (Clearance: {result['clearance']})")
    print("="*80)

    print("\n--- RETRIEVAL STEP & EXPLANATIONS ---")
    for explanation in result["retrieval_explanations"]:
        print(explanation)

    print("\n--- RETRIEVED CHUNKS ---")
    if not result["retrieved_chunks"]:
        print("[No authorized chunks returned]")
    for chunk in result["retrieved_chunks"]:
        print(f"- {chunk['id']} (Similarity: {chunk['similarity']:.2%}, Clearance: {chunk['clearance']})")
        print(f"  Snippet: {chunk['snippet']}")

    print("\n--- GENERATED RESPONSE ---")
    print(result["response"])

    print("\n--- JUDGE EVALUATION ---")
    eval_res = result["evaluation"]
    print(f"Faithfulness Score: {eval_res.get('faithfulness_score')}/5")
    print(f"Faithfulness Detail: {eval_res.get('faithfulness_reason')}")
    print(f"Completeness Score: {eval_res.get('completeness_score')}/5")
    print(f"Completeness Detail: {eval_res.get('completeness_reason')}")
    print(f"Approved for User: {eval_res.get('approved')}")
    print(f"Auditor Feedback: {eval_res.get('feedback')}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
