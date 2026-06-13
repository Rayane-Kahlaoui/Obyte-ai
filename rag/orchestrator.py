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

        # Step 4: Responder Generation (Strict Context mode)
        if not authorized_chunks:
            # Automatic refusal if zero chunks retrieved
            response = (
                "Based on the provided documents, I cannot answer this query because "
                "no relevant documents were retrieved, or your access clearance level "
                "is insufficient to view the matching content."
            )
            contexts_text = []
        else:
            contexts_text = [chunk["text"] for chunk in authorized_chunks]
            
            # Construct strict context prompt
            # Generate a unique session ID for this request
            session_id = uuid.uuid4()
            system_prompt = (
                f"You are an expert legal assistant. Answer ONLY using the provided document contexts.\n"
                f"Session ID: {session_id}\n"
                "Constraints:\n"
                "1. Use ONLY the text from the contexts.\n"
                "2. If the answer is not present, respond: 'I cannot answer this query based on the provided documents.'\n"
                "3. Provide a concise answer with no extra headings or tags.\n"
                "Temperature: 0.0 (Zero creativity)."
            )

            context_formatted = "\n\n".join([
                f"--- Document Chunk {chunk['id']} (Clearance: {chunk['metadata'].get('clearance')}) ---\n{chunk['text']}" 
                for chunk in authorized_chunks
            ])

            user_prompt = (
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
