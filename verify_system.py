import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to enable local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.config import CLEARANCE_LEVELS
from rag.agents.identity_agent import IdentityAgent
from rag.agents.retrieval_agent import RetrievalAgent
from rag.agents.judge_agent import JudgeAgent
from rag.orchestrator import RAGOrchestrator
from rag.mcp_server import query_legal_system, verify_access_permission, evaluate_response_grounding, orchestrator as mcp_orchestrator
from rag.llm_client import LLMClient

# Force verify_system to use mock backend to prevent rate-limiting or network issues during automated tests
mock_client = LLMClient(api_type="mock")
mcp_orchestrator.llm_client = mock_client
mcp_orchestrator.judge_agent.llm_client = mock_client
mcp_orchestrator.retrieval_agent.llm_client = mock_client
mcp_orchestrator.identity_agent.llm_client = mock_client

def test_identity_agent():
    print("Testing Identity Agent...")
    agent = IdentityAgent()
    
    # Test clearance mappings
    assert agent.get_user_clearance("alice") == "Confidential"
    assert agent.get_user_clearance("bob") == "Internal"
    assert agent.get_user_clearance("charlie") == "Public"
    assert agent.get_user_clearance("unknown_user") == "Public"
    assert agent.get_user_clearance("") == "Public"
    
    # Test authorization logic
    assert agent.is_authorized("Confidential", "Public") is True
    assert agent.is_authorized("Confidential", "Internal") is True
    assert agent.is_authorized("Confidential", "Confidential") is True
    
    assert agent.is_authorized("Internal", "Public") is True
    assert agent.is_authorized("Internal", "Internal") is True
    assert agent.is_authorized("Internal", "Confidential") is False
    
    assert agent.is_authorized("Public", "Public") is True
    assert agent.is_authorized("Public", "Internal") is False
    assert agent.is_authorized("Public", "Confidential") is False
    
    # Test verify_access structured response
    res = agent.verify_access("bob", "Confidential")
    assert res["authorized"] is False
    assert "DENIED" in res["explanation"]
    
    res_ok = agent.verify_access("alice", "Internal")
    assert res_ok["authorized"] is True
    assert "AUTHORIZED" in res_ok["explanation"]
    
    print("[PASS] Identity Agent tests passed.")

def test_judge_agent():
    print("Testing Judge Agent...")
    agent = JudgeAgent(llm_client=mock_client)
    
    # 1. Test Refusal Case (No context, compliant refusal)
    refusal_eval = agent.evaluate_response(
        query="What is compliance policy?",
        retrieved_contexts=[],
        response="I cannot answer this query based on the provided documents."
    )
    assert refusal_eval["approved"] is True
    assert refusal_eval["faithfulness_score"] == 5
    assert refusal_eval["completeness_score"] == 5
    
    # 2. Test Grounded Response Case
    contexts = ["The compliance policy requires annual cybersecurity training for all staff."]
    grounded_response = "[Local Mock LLM Response - Strict Mode] The compliance policy requires annual cybersecurity training for all staff."
    grounded_eval = agent.evaluate_response(
        query="What training is required?",
        retrieved_contexts=contexts,
        response=grounded_response
    )
    assert grounded_eval["approved"] is True
    assert grounded_eval["faithfulness_score"] == 5
    
    # 3. Test Hallucination Case (Adding outside details not in context)
    hallucinated_response = "[Local Mock LLM Response - Strict Mode] Staff must take training and also pay a $500 penalty fee if they fail."
    hallucinated_eval = agent.evaluate_response(
        query="What training is required?",
        retrieved_contexts=contexts,
        response=hallucinated_response
    )
    # The grounding ratio should be lower since 'penalty', 'fee', 'fail', '500' are not in context
    assert hallucinated_eval["faithfulness_score"] < 5
    
    print("[PASS] Judge Agent tests passed.")

def test_orchestrator_and_retrieval():
    print("Testing Orchestrator and Retrieval...")
    orchestrator = RAGOrchestrator(llm_client=mock_client)
    
    # Run a test query as Alice (Confidential)
    alice_res = orchestrator.process_query("alice", "document compliance", top_k=2)
    assert alice_res["clearance"] == "Confidential"
    assert len(alice_res["retrieved_chunks"]) <= 2
    
    # Run a test query as Charlie (Public)
    charlie_res = orchestrator.process_query("charlie", "document compliance", top_k=5)
    assert charlie_res["clearance"] == "Public"
    
    # Ensure Charlie's retrieved chunks do NOT contain Internal or Confidential clearances
    for chunk in charlie_res["retrieved_chunks"]:
        assert chunk["clearance"] == "Public"
        
    print("[PASS] Orchestrator and Retrieval tests passed.")

def test_mcp_tools():
    print("Testing MCP Tools Registration & Execution...")
    
    # Test verify_access_permission tool
    mcp_verify_denied = verify_access_permission("charlie", "Confidential")
    assert "DENIED" in mcp_verify_denied
    
    mcp_verify_ok = verify_access_permission("alice", "Public")
    assert "AUTHORIZED" in mcp_verify_ok
    
    # Test evaluate_response_grounding tool
    mcp_eval = evaluate_response_grounding(
        query="Is training annual?",
        retrieved_context="Training is required annually.",
        response="Yes, training is required annually."
    )
    assert "Approved: True" in mcp_eval
    assert "Faithfulness Score: 5/5" in mcp_eval
    
    # Test query_legal_system tool
    mcp_query_res = query_legal_system("charlie", "document creation")
    assert "charlie" in mcp_query_res
    assert "Clearance: Public" in mcp_query_res
    assert "Answer" in mcp_query_res
    
    print("[PASS] MCP Tools tests passed.")

if __name__ == "__main__":
    print("=========================================")
    print("Running Multi-Agent Legal RAG Verification Suite")
    print("=========================================")
    try:
        test_identity_agent()
        test_judge_agent()
        test_orchestrator_and_retrieval()
        test_mcp_tools()
        print("=========================================")
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=========================================")
    except AssertionError as e:
        print("\n[FAIL] TEST FAILURE:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
