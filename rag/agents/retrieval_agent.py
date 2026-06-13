import re
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from rag.agents.base import BaseAgent
from rag.config import (
    DATABASE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    MIN_SIMILARITY
)

class RetrievalAgent(BaseAgent):
    def __init__(self, llm_client=None):
        super().__init__(
            name="Retrieval Agent",
            role="Semantic Document Query & Explainability Engine",
            llm_client=llm_client
        )
        # Connect to local ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.emb_fn
        )

    def retrieve_and_explain(
        self,
        query: str,
        user_clearance: str,
        identity_agent,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Queries ChromaDB for relevant document chunks, filters them based on access authorization,
        and provides an explainable report of the retrieval process.
        """
        # Fetch more candidates to ensure we have top_k results after permission filtering
        candidate_count = top_k * 20  # broaden search to improve chance of finding relevant ambiguous chunk
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=candidate_count
            )
        except Exception as e:
            return {
                "chunks": [],
                "explanations": [f"Database query failed: {e}"],
                "all_attempts": []
            }

        # Check for empty database or no results
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return {
                "chunks": [],
                "explanations": ["No document chunks match the query in the database."],
                "all_attempts": []
            }

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        authorized_hits = []
        explanations = []
        all_attempts = []

        # Common stopwords to exclude from explainability word-overlap log
        stopwords = {
            "the", "and", "a", "of", "in", "to", "for", "is", "that", "on", "with", 
            "as", "by", "at", "an", "be", "this", "are", "from", "it", "or", "which",
            "le", "la", "les", "des", "du", "un", "une", "et", "en", "est", "dans", "pour"
        }

        # Extrapolate query keywords
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - stopwords

        for i in range(len(ids)):
            chunk_id = ids[i]
            chunk_text = documents[i]
            meta = metadatas[i]
            distance = distances[i]
            similarity = max(0.0, min(1.0, 1.0 - distance))  # Normalize cosine similarity
            # Apply minimum similarity threshold
            if similarity < MIN_SIMILARITY:
                # Treat as not authorized / insufficient relevance
                is_authorized = False
                explanation = (
                    f"[FILTERED OUT] Chunk '{chunk_id}' matched with {similarity:.2%} confidence (below threshold). "
                    f"Required clearance: '{resource_clearance}' (Access Denied: similarity below {MIN_SIMILARITY:.2f})."
                )
                explanations.append(explanation)
                continue
            
            resource_clearance = meta.get("clearance", "Public")
            
            # Verify access using the IdentityAgent
            is_authorized = identity_agent.is_authorized(user_clearance, resource_clearance)
            
            # Calculate keyword match for explainability
            doc_words = set(re.findall(r'\b\w{3,}\b', chunk_text.lower())) - stopwords
            matched_words = query_words.intersection(doc_words)
            keyword_context = ", ".join(list(matched_words)[:4]) if matched_words else "semantic connection"

            attempt_detail = {
                "id": chunk_id,
                "clearance": resource_clearance,
                "similarity": similarity,
                "authorized": is_authorized
            }
            all_attempts.append(attempt_detail)

            if is_authorized:
                explanation = (
                    f"[RETRIEVED] Chunk '{chunk_id}' (Doc {meta.get('doc_id')}) matched with "
                    f"{similarity:.2%} confidence. Required clearance: '{resource_clearance}' (Authorized). "
                    f"Overlapping concepts: [{keyword_context}]."
                )
                explanations.append(explanation)
                
                authorized_hits.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": meta,
                    "similarity": similarity
                })
            else:
                explanation = (
                    f"[FILTERED OUT] Chunk '{chunk_id}' matched with {similarity:.2%} confidence. "
                    f"Required clearance: '{resource_clearance}' (Access Denied: User level is '{user_clearance}')."
                )
                explanations.append(explanation)

        # After scanning all candidates, select the most relevant authorized chunks
        # Sort by similarity descending and keep the top_k most similar
        if authorized_hits:
            authorized_hits.sort(key=lambda c: c["similarity"], reverse=True)
            authorized_hits = authorized_hits[:top_k]

        return {
            "chunks": authorized_hits,
            "explanations": explanations,
            "all_attempts": all_attempts
        }
