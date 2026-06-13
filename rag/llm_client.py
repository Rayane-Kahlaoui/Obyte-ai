import os
import requests
from typing import Dict, Any, List
from rag.config import (
    LLM_API_TYPE,
    HF_API_URL,
    HF_API_MODEL,
    LOCAL_LLM_API_URL,
    LOCAL_LLM_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL
)

class LLMClient:
    def __init__(
        self,
        api_type: str = None,
        hf_model: str = HF_API_MODEL,
        hf_url: str = HF_API_URL,
        local_model: str = LOCAL_LLM_MODEL,
        local_url: str = LOCAL_LLM_API_URL,
        temperature: float = LLM_TEMPERATURE
    ):
        self.api_type = api_type or LLM_API_TYPE
        self.hf_model = hf_model
        self.hf_url = hf_url
        self.local_model = local_model
        self.local_url = local_url
        self.temperature = temperature
        self.token = os.environ.get("HF_TOKEN", "")
        # OpenRouter credentials
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        # DeepSeek V4 Flash model identifier for OpenRouter
        self.openrouter_model = "openrouter/free"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the LLM based on configured API type.
        """
        if self.api_type == "mock":
            return self._mock_generate(system_prompt, user_prompt)
        elif self.api_type == "local_api":
            return self._live_generate(self.local_url, self.local_model, system_prompt, user_prompt)
        elif self.api_type == "openrouter":
            return self._openrouter_generate(system_prompt, user_prompt)
        else:  # hf_inference default
            return self._hf_generate(system_prompt, user_prompt)

    def _hf_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Calls the Hugging Face Serverless Inference API.
        """
        headers = {
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        data = {
            "model": self.hf_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 800
        }
        
        try:
            url = f"{self.hf_url}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            error_msg = (
                f"Error communicating with Hugging Face Serverless Inference API at {self.hf_url}.\n"
                f"Details: {e}\n"
            )
            if not self.token:
                error_msg += (
                    "Warning: HF_TOKEN environment variable is not set. The Serverless API "
                    "requires authentication for regular access. Please run 'huggingface-cli login' "
                    "or export HF_TOKEN in your environment.\n"
                )
            try:
                if 'response' in locals() and hasattr(response, 'text'):
                    error_msg += f"Server response: {response.text}\n"
            except:
                pass
            return error_msg

    def _live_generate(self, api_url: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
        """
        Calls a local OpenAI-compatible API running locally (e.g. Ollama, llama.cpp server).
        """
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature
        }
        try:
            url = f"{api_url}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            return (
                f"Error communicating with local LLM server at {api_url}.\n"
                f"Details: {e}\n"
                f"Please ensure your local inference server (e.g., Ollama or llama.cpp) "
                f"is running and the model '{model_name}' is downloaded."
            )

    def _mock_generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Fallback mock generator parsing sentences from the context.
        """
        context = ""
        context_markers = ["<context>", "Context:", "CONTEXT:"]
        for marker in context_markers:
            if marker in user_prompt:
                start_idx = user_prompt.find(marker) + len(marker)
                end_marker = marker.replace("<", "</") if marker.startswith("<") else "\n\n"
                end_idx = user_prompt.find(end_marker, start_idx)
                if end_idx == -1:
                    context = user_prompt[start_idx:]
                else:
                    context = user_prompt[start_idx:end_idx]
                break
        
        context = context.strip()
        is_empty_context = (
            not context or 
            "[no document retrieved]" in context.lower() or 
            "empty context" in context.lower()
        )

        if is_empty_context:
            return (
                "Based on the provided documents, I cannot answer this query because "
                "no relevant documents were retrieved or the provided context is empty."
            )

        sentences = [s.strip() for s in context.split(".") if s.strip()]
        if len(sentences) > 0:
            extracted_fact = sentences[0]
            if len(sentences) > 1:
                extracted_fact += ". " + sentences[1]
            return (
                f"[Local Mock LLM Response - Strict Mode]\n"
                f"Based on the provided legal documentation, the following points apply:\n"
                f"1. {extracted_fact}.\n"
                f"This answer is strictly derived from the retrieved documents to ensure zero hallucination."
            )
        
        return "Based on the provided documents, I cannot find sufficient detail to formulate an accurate response."

    def _openrouter_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response via OpenRouter (DeepSeek V4 Flash)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_key}"
        }
        data = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 800
        }
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            # OpenRouter returns similar structure to OpenAI
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            err = f"Error communicating with OpenRouter API at {OPENROUTER_API_URL}.\nDetails: {e}\n"
            if not self.openrouter_key:
                err += "Warning: OPENROUTER_API_KEY environment variable is not set.\n"
            try:
                if 'response' in locals() and hasattr(response, 'text'):
                    err += f"Server response: {response.text}\n"
            except:
                pass
            return err
