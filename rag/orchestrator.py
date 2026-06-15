import uuid
from typing import Dict, Any

from rag.llm_client import LLMClient
from rag.agents.identity_agent import IdentityAgent
from rag.agents.retrieval_agent import RetrievalAgent
from rag.agents.judge_agent import JudgeAgent

class RAGOrchestrator:
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.identity_agent = IdentityAgent(llm_client=self.llm_client)
        self.retrieval_agent = RetrievalAgent(llm_client=self.llm_client)
        self.judge_agent = JudgeAgent(llm_client=self.llm_client)

    def process_query(self, username: str, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Coordinates the entire RAG pipeline:
        1. Checks user clearance.
        2. Retrieves relevant document chunks.
        3. Excludes unauthorized results.
        4. Generates a response strictly grounded in context.
        5. Audits the final answer via the Judge Agent.
        """
        # Step 1: Verification & Permission Assertion
        user_clearance = self.identity_agent.get_user_clearance(username)
        
        # Step 2 & 3: Retrieval and Filtering
        retrieval_result = self.retrieval_agent.retrieve_and_explain(
            query=query,
            user_clearance=user_clearance,
            identity_agent=self.identity_agent,
            top_k=top_k
        )

        authorized_chunks = retrieval_result["chunks"]
        retrieval_explanations = retrieval_result["explanations"]

        # Check if any matching chunks were filtered out due to unauthorized clearance
        denied_attempts = [
            attempt for attempt in retrieval_result.get("all_attempts", [])
            if not attempt.get("authorized", True)
        ]
        has_denied_access = len(denied_attempts) > 0

        # Step 4: Responder Generation (Strict Context mode)
        if not authorized_chunks:
            # Automatic refusal if zero chunks retrieved
            if has_denied_access:
                response = (
                    "Based on the provided documents, I cannot answer this query because "
                    "matching documents exist but your access clearance level is insufficient to view them.\n"
                    "Notice: Some matching documents were filtered out due to insufficient access clearance level."
                )
            else:
                response = (
                    "Based on the provided documents, I cannot answer this query because "
                    "no relevant documents were retrieved."
                )
            contexts_text = []
        else:
            contexts_text = [chunk["text"] for chunk in authorized_chunks]
            
            # Construct strict context prompt
            # Generate a unique session ID for this request
            session_id = uuid.uuid4()
            system_prompt = (
                f"You are Orbyte AI, an advanced expert Legal Document Assistant and Compliance Auditor.\n"
                f"Your primary role is to answer user queries with absolute accuracy, relying EXCLUSIVELY on the provided document context below.\n"
                f"Session Reference ID: {session_id}\n\n"
                "CRITICAL INSTRUCTIONS & CONSTRAINTS:\n"
                "1. STRICT TRUTH AND GROUNDING: You must answer the query using ONLY the factual statements directly present in the context. "
                "Do NOT assume, extrapolate, or bring in external knowledge. If the context does not contain the answer, you must respond exactly with: "
                "'I cannot answer this query based on the provided documents.'\n"
                "2. NO CREATIVITY OR HALLUCINATION: Maintain a zero-creativity threshold. Any facts, dates, percentages, or terms not explicitly detailed in the context must be omitted.\n"
                "3. ACCESS CLEARANCE & SECURITY NOTICE: If you are informed in the prompt that some matching documents were filtered out due to insufficient user access clearance, "
                "you MUST explicitly include this statement in your response: "
                "'Notice: Some matching documents were filtered out due to insufficient access clearance level.' "
                "Combine this explanation clearly in your response if the retrieved context is missing information to fully answer.\n"
                "4. FORMATTING: Provide a clean, direct, and concise legal response. Do not use markdown headers, summary wrappers, or HTML tags. Start directly with the response text."
            )

            context_formatted = "\n\n".join([
                f"--- Document Chunk {chunk['id']} (Clearance: {chunk['metadata'].get('clearance')}) ---\n{chunk['text']}" 
                for chunk in authorized_chunks
            ])

            user_prompt = ""
            if has_denied_access:
                user_prompt += "[System Notification: Some matching documents were filtered out from the context below because the user does not have permission/clearance to view them.]\n\n"
            
            user_prompt += (
                f"<context>\n{context_formatted}\n</context>\n\n"
                f"Query: {query}\n"
            )

            response = self.llm_client.generate(system_prompt, user_prompt)

        # Step 5: Judge Audit
        evaluation = self.judge_agent.evaluate_response(
            query=query,
            retrieved_contexts=contexts_text,
            response=response
        )

        return {
            "query": query,
            "username": username,
            "clearance": user_clearance,
            "response": response,
            "retrieved_chunks": [
                {
                    "id": c["id"],
                    "similarity": c["similarity"],
                    "clearance": c["metadata"].get("clearance"),
                    "snippet": c["text"][:120] + "..." if len(c["text"]) > 120 else c["text"]
                } for c in authorized_chunks
            ],
            "retrieval_explanations": retrieval_explanations,
            "evaluation": evaluation,
            "approved": evaluation["approved"]
        }
