import sys
import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory of 'api' folder to sys.path to resolve 'rag' module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.orchestrator import RAGOrchestrator
from rag.agents.identity_agent import IdentityAgent
from rag.agents.judge_agent import JudgeAgent
from rag.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, LLM_API_TYPE

app = FastAPI(
    title="Orbyte RAG API",
    description="REST API for the Orbyte Sovereign Multi-Agent Legal RAG System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared orchestrator and agents instances
orchestrator = RAGOrchestrator()
identity_agent = IdentityAgent()
judge_agent = JudgeAgent()

# --- Schemas ---

class QueryRequest(BaseModel):
    username: str = Field(..., description="Identifiant de l'utilisateur", example="bob")
    query: str    = Field(..., description="Question en langage naturel", example="What is compliance policy?")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Nombre de chunks a recuperer")

class ChunkResult(BaseModel):
    chunk_id:   str
    similarity: float
    clearance:  str
    snippet:    str

class AuditResult(BaseModel):
    faithfulness_score:  int
    completeness_score:  int
    approved:            bool
    feedback:            str

class QueryResponse(BaseModel):
    query:           str
    username:        str
    clearance:       str
    response:        str
    approved:        bool
    chunks_used:     List[ChunkResult]
    chunks_filtered: int
    evaluation:      AuditResult
    processing_time: float

class VerifyResponse(BaseModel):
    username:           str
    user_clearance:     str
    resource_clearance: str
    authorized:         bool
    explanation:        str

class AuditRequest(BaseModel):
    query:              str = Field(..., example="What training is required?")
    retrieved_contexts: List[str] = Field(..., example=["The compliance policy requires annual training."])
    response:           str = Field(..., example="Annual training is required.")

class HealthResponse(BaseModel):
    status:             str
    chroma_collection:  str
    chunk_count:        int
    embedding_model:    str
    llm_mode:           str
    llm_model:          str
    version:            str

# --- Routes ---

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_query(payload: QueryRequest):
    """
    Main RAG pipeline entrypoint. Coordinates user authentication,
    document retrieval, safety filtering, LLM generation, and factual auditing.
    """
    start_time = time.time()
    try:
        # Run orchestrator
        result = orchestrator.process_query(
            username=payload.username,
            query=payload.query,
            top_k=payload.top_k or 3
        )
        
        # Calculate chunks_filtered based on explanations
        explanations = result.get("retrieval_explanations", [])
        chunks_filtered = sum(1 for exp in explanations if "[FILTERED OUT]" in exp)
        
        # Format response
        chunks_used = [
            ChunkResult(
                chunk_id=c["id"],
                similarity=c["similarity"],
                clearance=c["clearance"],
                snippet=c["snippet"]
            ) for c in result.get("retrieved_chunks", [])
        ]
        
        # Check if authorized chunks were empty (potential full security refusal or no matches)
        if not chunks_used:
            # Check if this query returned only filtered chunks due to access limits
            has_filtered_chunks = any("Access Denied" in exp for exp in explanations)
            if has_filtered_chunks:
                # User has insufficient clearance to access matching documents
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: no authorized documents found for user '{payload.username}' matching this query."
                )
        
        eval_data = result.get("evaluation", {})
        evaluation = AuditResult(
            faithfulness_score=eval_data.get("faithfulness_score", 0),
            completeness_score=eval_data.get("completeness_score", 0),
            approved=eval_data.get("approved", False),
            feedback=eval_data.get("feedback", "No feedback available")
        )
        
        processing_time = round(time.time() - start_time, 3)
        
        return QueryResponse(
            query=result["query"],
            username=result["username"],
            clearance=result["clearance"],
            response=result["response"],
            approved=result["approved"],
            chunks_used=chunks_used,
            chunks_filtered=chunks_filtered,
            evaluation=evaluation,
            processing_time=processing_time
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/auth/verify", response_model=VerifyResponse, status_code=status.HTTP_200_OK)
async def verify_auth(
    user: str = Query(..., description="Nom d'utilisateur"),
    document_clearance: str = Query(..., description="Niveau de clearance requis")
):
    """
    Verifies if a user is authorized to access a specific document clearance level.
    """
    try:
        res = identity_agent.verify_access(user, document_clearance)
        return VerifyResponse(
            username=res["username"],
            user_clearance=res["user_clearance"],
            resource_clearance=res["resource_clearance"],
            authorized=res["authorized"],
            explanation=res["explanation"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing access check: {str(e)}"
        )

@app.post("/api/v1/audit", response_model=AuditResult, status_code=status.HTTP_200_OK)
async def audit_response(payload: AuditRequest):
    """
    Audits a response for factual grounding and completeness against retrieved contexts.
    """
    try:
        res = judge_agent.evaluate_response(
            query=payload.query,
            retrieved_contexts=payload.retrieved_contexts,
            response=payload.response
        )
        return AuditResult(
            faithfulness_score=res["faithfulness_score"],
            completeness_score=res["completeness_score"],
            approved=res["approved"],
            feedback=res["feedback"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing audit: {str(e)}"
        )

@app.get("/api/v1/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def get_health():
    """
    Checks system health, databases connectivity, and retrieval configuration.
    """
    try:
        chunk_count = orchestrator.retrieval_agent.collection.count()
        llm_model = orchestrator.llm_client.model_name if hasattr(orchestrator.llm_client, "model_name") else "unknown"
        if not llm_model or llm_model == "unknown":
            # Fallback to local configs
            from rag.config import LOCAL_LLM_MODEL
            llm_model = LOCAL_LLM_MODEL
            
        return HealthResponse(
            status="healthy",
            chroma_collection=COLLECTION_NAME,
            chunk_count=chunk_count,
            embedding_model=EMBEDDING_MODEL_NAME,
            llm_mode=LLM_API_TYPE,
            llm_model=llm_model,
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database or system connection unhealthy: {str(e)}"
        )
