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
    OPENROUTER_API_URL,
    OPENROUTER_MODEL
)

class LLMClient:
    # Class-level caches to share loaded models between instances
    _llama_model = None
    _transformers_model = None
    _transformers_tokenizer = None

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
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "") or OPENROUTER_API_KEY
        # Configure model identifier from environment or config.py default
        self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "") or OPENROUTER_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the LLM based on configured API type.
        """
        if self.api_type == "mock":
            return self._mock_generate(system_prompt, user_prompt)
        elif self.api_type == "local_api":
            return self._live_generate(self.local_url, self.local_model, system_prompt, user_prompt)
        elif self.api_type == "local_python":
            return self._local_python_generate(system_prompt, user_prompt)
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

    def _local_python_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Runs self-contained local Python inference via llama-cpp or transformers."""
        # Import settings lazily
        from rag.config import (
            LOCAL_INFERENCE_TYPE,
            LOCAL_HF_MODEL_ID,
            LOCAL_HF_GGUF_FILENAME,
            LOCAL_MODEL_PATH
        )
        
        if LOCAL_INFERENCE_TYPE == "llama_cpp":
            return self._generate_with_llama_cpp(
                LOCAL_HF_MODEL_ID, LOCAL_HF_GGUF_FILENAME, LOCAL_MODEL_PATH, system_prompt, user_prompt
            )
        elif LOCAL_INFERENCE_TYPE == "transformers":
            return self._generate_with_transformers(
                LOCAL_HF_MODEL_ID, LOCAL_MODEL_PATH, system_prompt, user_prompt
            )
        else:
            return f"Unsupported LOCAL_INFERENCE_TYPE: {LOCAL_INFERENCE_TYPE}"

    def _generate_with_llama_cpp(
        self, hf_repo: str, gguf_filename: str, model_path: str, system_prompt: str, user_prompt: str
    ) -> str:
        if LLMClient._llama_model is None:
            import os
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            
            # Resolve actual model file path
            if model_path and os.path.exists(model_path):
                resolved_path = model_path
            else:
                actual_filename = gguf_filename
                if actual_filename == "*q4_k_m.gguf":
                    actual_filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
                
                print(f"\n[Local Engine] Downloading/Checking GGUF model '{actual_filename}' from repo '{hf_repo}'...")
                try:
                    resolved_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=actual_filename,
                        repo_type="model"
                    )
                except Exception as e:
                    print(f"Error downloading {actual_filename} from {hf_repo}: {e}. Trying fallback bartowski/Qwen2.5-1.5B-Instruct-GGUF...")
                    try:
                        resolved_path = hf_hub_download(
                            repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
                            filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
                            repo_type="model"
                        )
                    except Exception as fallback_err:
                        return f"Failed to download GGUF model: {e}\nFallback error: {fallback_err}"
            
            print(f"[Local Engine] Loading Llama model from {resolved_path}...")
            LLMClient._llama_model = Llama(
                model_path=resolved_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            print("[Local Engine] Llama model loaded successfully.")
            
        try:
            response = LLMClient._llama_model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=800
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error executing local llama-cpp generation: {e}"

    def _generate_with_transformers(
        self, model_id: str, model_path: str, system_prompt: str, user_prompt: str
    ) -> str:
        if LLMClient._transformers_model is None:
            import os
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            actual_model = model_path if (model_path and os.path.exists(model_path)) else model_id
            if not actual_model or actual_model.endswith("-GGUF"):
                actual_model = "Qwen/Qwen2.5-1.5B-Instruct"
                
            print(f"\n[Local Engine] Loading Transformers model/tokenizer for '{actual_model}'...")
            LLMClient._transformers_tokenizer = AutoTokenizer.from_pretrained(actual_model)
            LLMClient._transformers_model = AutoModelForCausalLM.from_pretrained(
                actual_model,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            print("[Local Engine] Transformers model loaded successfully.")
            
        try:
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = LLMClient._transformers_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = LLMClient._transformers_tokenizer([text], return_tensors="pt").to(LLMClient._transformers_model.device)
            with torch.no_grad():
                generated_ids = LLMClient._transformers_model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0.0
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return LLMClient._transformers_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            return f"Error executing local transformers generation: {e}"
