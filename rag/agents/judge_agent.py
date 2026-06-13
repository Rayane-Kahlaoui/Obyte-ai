import json
import re
from typing import List, Dict, Any
from rag.agents.base import BaseAgent

class JudgeAgent(BaseAgent):
    def __init__(self, llm_client=None):
        super().__init__(
            name="Response Judge Agent",
            role="Factual Grounding & Retrieval Quality Auditor",
            llm_client=llm_client
        )

    def evaluate_response(
        self,
        query: str,
        retrieved_contexts: List[str],
        response: str
    ) -> Dict[str, Any]:
        """
        Evaluates the generated response against retrieved contexts.
        Returns score metrics, descriptions, and approval status.
        """
        # If mock LLM is active in the client, we run a smart rule-based local judge evaluation.
        # Otherwise, we ask the local LLM to perform the evaluation.
        if self.llm_client.api_type == "mock":
            return self._mock_evaluate(query, retrieved_contexts, response)
        return self._live_evaluate(query, retrieved_contexts, response)

    def _mock_evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        response: str
    ) -> Dict[str, Any]:
        """
        Performs a deterministic, rule-based verification of response quality.
        - Refusals get high faithfulness.
        - Grounded mock responses get high ratings.
        - Hallucinations (e.g., outside words or invalid contexts) are caught and penalized.
        """
        # Case 1: The system refused to answer because of empty context
        is_refusal = (
            "cannot answer" in response.lower() or 
            "no relevant documents" in response.lower() or 
            "context is empty" in response.lower()
        )
        
        if is_refusal:
            if not retrieved_contexts or len(retrieved_contexts) == 0:
                # Correctly refused empty database or unauthorized query
                return {
                    "faithfulness_score": 5,
                    "faithfulness_reason": "The system correctly refused to answer since no context was retrieved (Authorized context-only behavior).",
                    "completeness_score": 5,
                    "completeness_reason": "The query was answered with a correct and compliant refusal.",
                    "approved": True,
                    "feedback": "Perfect refusal behavior. Avoided hallucination on empty context."
                }
            else:
                # Refused even though context exists - maybe incomplete matching
                return {
                    "faithfulness_score": 5,
                    "faithfulness_reason": "System refused to answer. Factual grounding is correct (no false statements made).",
                    "completeness_score": 2,
                    "completeness_reason": "System refused to answer even though context was available.",
                    "approved": False,
                    "feedback": "The model refused to answer. Consider refining query matching."
                }

        # Case 2: We have a generated response. Let's verify grounding.
        # Merge all retrieved contexts into a single block
        full_context = " ".join(retrieved_contexts).lower()
        
        # Strip mock tags for clean analysis
        clean_response = response.replace("[Local Mock LLM Response - Strict Mode]", "")
        clean_response = re.sub(r'Based on the provided legal documentation, the following points apply:', '', clean_response)
        clean_response = re.sub(r'\d+\.', '', clean_response)
        clean_response = clean_response.strip().lower()

        # Split response into keywords (ignoring short/stop words)
        words_in_response = set(re.findall(r'\b\w{4,}\b', clean_response))
        words_in_context = set(re.findall(r'\b\w{4,}\b', full_context))
        
        # Stopwords specific to mock sentences
        mock_stopwords = {"based", "provided", "legal", "documentation", "following", "points", "apply", "answer", "strictly", "derived", "retrieved", "documents", "ensure", "hallucination"}
        words_in_response = words_in_response - mock_stopwords

        if not words_in_response:
            return {
                "faithfulness_score": 5,
                "faithfulness_reason": "Response does not contain verifiable claims.",
                "completeness_score": 3,
                "completeness_reason": "Response is too short to fully answer.",
                "approved": False,
                "feedback": "Response is empty or lacks substantive statements."
            }

        # Calculate grounding ratio
        grounded_words = words_in_response.intersection(words_in_context)
        grounding_ratio = len(grounded_words) / len(words_in_response)

        # Rate Faithfulness (1 to 5) based on ratio
        if grounding_ratio >= 0.8:
            faithfulness_score = 5
            faithfulness_reason = "100% grounded in the retrieved documents. Every major claim overlaps with source text."
        elif grounding_ratio >= 0.5:
            faithfulness_score = 4
            faithfulness_reason = "Highly grounded in retrieved documents with minor rephrasing."
        elif grounding_ratio >= 0.3:
            faithfulness_score = 3
            faithfulness_reason = "Moderately grounded, but contains some unverified/outside vocabulary."
        else:
            faithfulness_score = 2
            faithfulness_reason = "Warning: Low grounding. Response contains substantial terminology not present in source context."

        # Completeness checks: Did we answer the query terms?
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        response_terms = set(re.findall(r'\b\w{4,}\b', response.lower()))
        query_matches = query_terms.intersection(response_terms)
        
        if len(query_terms) == 0:
            completeness_score = 5
            completeness_reason = "Query was too short to analyze."
        else:
            match_ratio = len(query_matches) / len(query_terms)
            if match_ratio >= 0.6:
                completeness_score = 5
                completeness_reason = "Response touches on all key aspects of the user's query."
            elif match_ratio >= 0.3:
                completeness_score = 4
                completeness_reason = "Response addresses the query adequately."
            else:
                completeness_score = 3
                completeness_reason = "Response addresses the query partially but may miss context."

        approved = faithfulness_score >= 4 and completeness_score >= 4

        return {
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "completeness_score": completeness_score,
            "completeness_reason": completeness_reason,
            "approved": approved,
            "feedback": "Response approved for display." if approved else "Response flags low quality or lack of grounding. Re-query recommended."
        }

    def _live_evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        response: str
    ) -> Dict[str, Any]:
        """
        Asks the local LLM to audit the generated response.
        """
        system_prompt = (
            "You are an expert legal auditor checking a RAG system's output.\n"
            "You must evaluate the response strictly on the retrieved source contexts.\n"
            "Provide your assessment as a single JSON object with these keys:\n"
            "{\n"
            '  "faithfulness_score": int (1-5),\n'
            '  "faithfulness_reason": "explanation of grounding",\n'
            '  "completeness_score": int (1-5),\n'
            '  "completeness_reason": "explanation of query alignment",\n'
            '  "approved": bool (true if both scores >= 4),\n'
            '  "feedback": "actionable feedback"\n'
            "}\n"
            "Do not include any other conversational text. Return only the JSON object."
        )

        user_prompt = (
            f"=== User Query ===\n{query}\n\n"
            f"=== Retrieved Contexts ===\n" + "\n---\n".join(retrieved_contexts) + "\n\n"
            f"=== Generated Response ===\n{response}\n\n"
            f"Evaluate the grounding and completeness of this response."
        )

        llm_output = self.llm_client.generate(system_prompt, user_prompt)

        # Parse JSON output from LLM
        try:
            # Clean up markdown formatting if the LLM wrapped it in ```json ... ```
            cleaned_output = llm_output.strip()
            if cleaned_output.startswith("```"):
                cleaned_output = re.sub(r"^```(?:json)?\n", "", cleaned_output)
                cleaned_output = re.sub(r"\n```$", "", cleaned_output)
            
            result = json.loads(cleaned_output)
            return result
        except Exception as e:
            # Fallback if LLM output fails to parse
            return {
                "faithfulness_score": 1,
                "faithfulness_reason": f"Failed to parse Judge LLM response. Raw output: {llm_output}",
                "completeness_score": 1,
                "completeness_reason": f"Error: {e}",
                "approved": False,
                "feedback": "Regenerate query; system auditor evaluation failed."
            }
        
        return result
