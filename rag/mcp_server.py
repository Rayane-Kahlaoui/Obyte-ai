import sys
import os
from mcp.server.fastmcp import FastMCP

# Add parent directory to path to enable local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.orchestrator import RAGOrchestrator
from rag.config import CLEARANCE_LEVELS

# Create the FastMCP Server instance
mcp = FastMCP("Legal-RAG-System")

# Initialize orchestrator
orchestrator = RAGOrchestrator()

@mcp.tool()
def query_legal_system(user: str, query: str) -> str:
    """
    Query the legal RAG multi-agent database.
    This runs the full orchestration pipeline: user authentication/clearance assertion, 
    semantic search matching, answer generation, and judge audit.
    
    Args:
        user: Username of the user querying (e.g. alice, bob, charlie).
        query: The question or search term.
    """
    try:
        result = orchestrator.process_query(user, query)
        
        # Structure clear, human-readable output
        output = [
            f"=== MULTI-AGENT RESPONSE REPORT ===",
            f"Query: '{result['query']}'",
            f"User Profile: '{result['username']}' (Clearance: {result['clearance']})",
            "",
            "--- Retrieval Explanations & Clearance Filtering ---"
        ]
        
        for exp in result["retrieval_explanations"]:
            output.append(exp)
            
        output.append("")
        output.append("--- Retrieved Chunks ---")
        if not result["retrieved_chunks"]:
            output.append("[No authorized documents accessed]")
        for chunk in result["retrieved_chunks"]:
            output.append(f"- {chunk['id']} (Similarity: {chunk['similarity']:.2%}, Clearance: {chunk['clearance']})")
            
        output.append("")
        output.append("--- Answer ---")
        output.append(result["response"])
        
        eval_res = result["evaluation"]
        output.append("")
        output.append("--- Auditor Factual Grounding Evaluation ---")
        output.append(f"Faithfulness Score: {eval_res.get('faithfulness_score')}/5 ({eval_res.get('faithfulness_reason')})")
        output.append(f"Completeness Score: {eval_res.get('completeness_score')}/5 ({eval_res.get('completeness_reason')})")
        output.append(f"Factual Audit Approved: {eval_res.get('approved')}")
        output.append(f"Auditor Feedback: {eval_res.get('feedback')}")
        output.append("="*40)
        
        return "\n".join(output)
    except Exception as e:
        return f"Error executing RAG process: {e}"

@mcp.tool()
def verify_access_permission(user: str, document_clearance: str) -> str:
    """
    Verifies if a specific user has permissions to access a document of a given clearance level.
    
    Args:
        user: Username of the user (e.g. alice, bob, charlie).
        document_clearance: The security clearance tag of the document (Public, Internal, Confidential).
    """
    try:
        report = orchestrator.identity_agent.verify_access(user, document_clearance)
        return report["explanation"]
    except Exception as e:
        return f"Error during verification: {e}"

@mcp.tool()
def evaluate_response_grounding(query: str, retrieved_context: str, response: str) -> str:
    """
    Directly runs the Response Judge Agent to evaluate how faithful and complete a generated 
    response is compared to the retrieved source text.
    
    Args:
        query: The original question asked.
        retrieved_context: The reference text the model should have used.
        response: The generated answer to verify.
    """
    try:
        contexts = [retrieved_context] if retrieved_context.strip() else []
        eval_res = orchestrator.judge_agent.evaluate_response(query, contexts, response)
        
        output = [
            "=== Judge Audit ===",
            f"Faithfulness Score: {eval_res.get('faithfulness_score')}/5 - {eval_res.get('faithfulness_reason')}",
            f"Completeness Score: {eval_res.get('completeness_score')}/5 - {eval_res.get('completeness_reason')}",
            f"Approved: {eval_res.get('approved')}",
            f"Feedback: {eval_res.get('feedback')}"
        ]
        return "\n".join(output)
    except Exception as e:
        return f"Error during evaluation: {e}"

@mcp.resource("clearance://hierarchy")
def get_clearance_hierarchy() -> str:
    """
    Exposes the security clearance hierarchy mapping.
    """
    hierarchy = " -> ".join(CLEARANCE_LEVELS)
    return f"Clearance Hierarchy levels: {hierarchy}\nMock Users clearance mappings: alice=Confidential, bob=Internal, charlie=Public."

if __name__ == "__main__":
    mcp.run()
