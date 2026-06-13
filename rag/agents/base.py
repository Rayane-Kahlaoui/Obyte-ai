from rag.llm_client import LLMClient

class BaseAgent:
    def __init__(self, name: str, role: str, llm_client: LLMClient = None):
        self.name = name
        self.role = role
        self.llm_client = llm_client or LLMClient()

    def __repr__(self):
        return f"Agent(name={self.name}, role={self.role})"
