import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data"

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Ensure directories exist
DATABASE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Dataset Configurations
DATASET_NAME = "Shekswess/legal-documents"
COLLECTION_NAME = "legal_documents"

# Embedding Configurations
# We use all-MiniLM-L6-v2 as a robust, lightweight local embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Semantic Chunking Configurations
# Threshold controls sensitivity to semantic shifts.
# Lower percentile (e.g., 85) = more splits (smaller chunks)
# Higher percentile (e.g., 95) = fewer splits (larger chunks)
SEMANTIC_SPLIT_PERCENTILE = 90
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 2000

# Minimum similarity (cosine) required for a chunk to be considered relevant
MIN_SIMILARITY = 0.0

# LLM Inference Configurations
# Options: "hf_inference" (HuggingFace), "local_api" (Ollama/local), "openrouter" (OpenRouter), "mock"
LLM_API_TYPE = os.environ.get("LLM_API_TYPE", "openrouter")
# OpenRouter settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# HuggingFace fallback (still kept for possible future use)
HF_API_MODEL = os.environ.get("HF_API_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_API_MODEL}"
# Base orchestrator system prompt structure (reference)
system_prompt = (
    "You are Orbyte AI, an advanced expert Legal Document Assistant and Compliance Auditor.\n"
    "Your primary role is to answer user queries with absolute accuracy, relying EXCLUSIVELY on the provided document context below.\n"
    "Session Reference ID: {session_id}\n\n"
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
# Local API configuration (Ollama/llama.cpp server)
LOCAL_LLM_API_URL = os.environ.get("LOCAL_LLM_API_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.2:1b")

LLM_TEMPERATURE = 0.0  # Zero temperature for deterministic, hallucination-free RAG responses

# Mock clearances for testing Identity Agent
MOCK_USER_CLEARANCE = {
    "alice": "Confidential",  # High access level
    "bob": "Internal",       # Medium access level
    "charlie": "Public",     # Standard access level
}

# Clearance levels hierarchy (higher index = higher access clearance)
CLEARANCE_LEVELS = ["Public", "Internal", "Confidential"]
