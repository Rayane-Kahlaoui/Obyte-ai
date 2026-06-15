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


from rag.orchestrator import RAGOrchestrator
from rag.llm_client import LLMClient
from rag.config import LLM_API_TYPE, OPENROUTER_MODEL, LOCAL_LLM_MODEL, LOCAL_LLM_API_URL

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

    if args.query:
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
    else:
        print("\n" + "="*80)
        print("         WELCOME TO ORBYTE AI - MULTI-AGENT LEGAL RAG SYSTEM CLI")
        print("="*80)
        print("Please choose your LLM Backend Mode for this session:")
        print(f"  [1] OpenRouter (Model: {OPENROUTER_MODEL})")
        print(f"  [2] Local Ollama (Model: {LOCAL_LLM_MODEL} at {LOCAL_LLM_API_URL})")
        print("  [3] Hugging Face Serverless API (Qwen/Qwen2.5-72B-Instruct)")
        print("  [4] Local Mock Backend (offline rule-based generation)")
        print("-" * 80)

        selected_mode = args.mode
        while True:
            try:
                choice = input("Select backend mode [1-4] (default 1): ").strip()
                if not choice:
                    selected_mode = "openrouter"
                    break
                if choice == "1":
                    selected_mode = "openrouter"
                    break
                elif choice == "2":
                    selected_mode = "local"
                    break
                elif choice == "3":
                    selected_mode = "hf"
                    break
                elif choice == "4":
                    selected_mode = "mock"
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return

        # Select User Clearance Level
        print("-" * 80)
        print("Please choose your Active User (Clearance Level) for this session:")
        print("  [1] alice   (Clearance: Confidential - Full access)")
        print("  [2] bob     (Clearance: Internal - Medium access)")
        print("  [3] charlie (Clearance: Public - Standard access)")
        print("-" * 80)

        selected_user = args.user
        while True:
            try:
                user_choice = input("Select user [1-3] (default 3): ").strip()
                if not user_choice:
                    selected_user = "charlie"
                    break
                if user_choice == "1":
                    selected_user = "alice"
                    break
                elif user_choice == "2":
                    selected_user = "bob"
                    break
                elif user_choice == "3":
                    selected_user = "charlie"
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 3.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return

        # Re-initialize client & orchestrator with chosen backend
        mode = api_map[selected_mode]
        llm_client = LLMClient(api_type=mode)
        orchestrator = RAGOrchestrator(llm_client=llm_client)

        print("\n" + "="*80)
        print(f"Active User: {selected_user}")
        print(f"LLM Mode   : {selected_mode} ({mode})")
        print(f"Top K Chunks: {args.top_k}")
        print("\nCommands:")
        print("  /user <username> : Switch active user (e.g. /user alice)")
        print("  /mode <mode>     : Switch LLM mode (hf, local, mock, openrouter)")
        print("  /top-k <num>     : Switch retrieval count")
        print("  /exit or /quit   : Exit interactive session")
        print("  /help            : Show this message again")
        print("="*80 + "\n")

        current_user = selected_user
        current_mode = selected_mode
        current_top_k = args.top_k

        while True:
            try:
                # Prompt showing active settings
                prompt_str = f"[{current_user} @ {current_mode}] > "
                user_input = input(prompt_str).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                if cmd in ["/exit", "/quit"]:
                    print("Goodbye!")
                    break
                elif cmd == "/help":
                    print("\nCommands:")
                    print("  /user <username> : Switch active user (alice, bob, charlie)")
                    print("  /mode <mode>     : Switch LLM mode (hf, local, mock, openrouter)")
                    print("  /top-k <num>     : Switch retrieval count")
                    print("  /exit or /quit   : Exit interactive session\n")
                    continue
                elif cmd == "/user":
                    if len(parts) < 2:
                        print("Error: Please specify a username (e.g. /user alice)")
                    else:
                        current_user = parts[1].lower()
                        print(f"Switched user to: {current_user}")
                    continue
                elif cmd == "/top-k":
                    if len(parts) < 2:
                        print("Error: Please specify a number")
                    else:
                        try:
                            current_top_k = int(parts[1])
                            print(f"Switched top-k to: {current_top_k}")
                        except ValueError:
                            print("Error: top-k must be an integer")
                    continue
                elif cmd == "/mode":
                    if len(parts) < 2:
                        print("Error: Please specify a mode (hf, local, mock, openrouter)")
                    else:
                        target_mode = parts[1].lower()
                        if target_mode in api_map:
                            current_mode = target_mode
                            # Reinitialize orchestrator and llm_client
                            mode = api_map[current_mode]
                            llm_client = LLMClient(api_type=mode)
                            orchestrator = RAGOrchestrator(llm_client=llm_client)
                            print(f"Switched LLM mode to: {current_mode} ({mode})")
                        else:
                            print(f"Error: Unknown mode '{target_mode}'. Choose from: {', '.join(api_map.keys())}")
                    continue
                else:
                    print(f"Unknown command: {cmd}. Type /help for assistance.")
                    continue

            # Run query
            print(f"\nProcessing query as user: {current_user}...")
            try:
                result = orchestrator.process_query(current_user, user_input, top_k=current_top_k)
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
            except Exception as e:
                print(f"\n[ERROR] An error occurred during processing: {e}\n")

if __name__ == "__main__":
    main()
