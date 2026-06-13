from rag.agents.base import BaseAgent
from rag.config import MOCK_USER_CLEARANCE, CLEARANCE_LEVELS

class IdentityAgent(BaseAgent):
    def __init__(self, llm_client=None):
        super().__init__(
            name="Identity Agent",
            role="User Authentication & Permission Guardrails",
            llm_client=llm_client
        )

    def get_user_clearance(self, username: str) -> str:
        """
        Determines the clearance level of the user. Defaults to 'Public' if user is unrecognized.
        """
        if not username:
            return "Public"
        return MOCK_USER_CLEARANCE.get(username.lower(), "Public")

    def is_authorized(self, user_clearance: str, resource_clearance: str) -> bool:
        """
        Checks if the user's clearance level is sufficient to access the resource.
        Clearance hierarchy is: Public < Internal < Confidential.
        """
        try:
            user_level_idx = CLEARANCE_LEVELS.index(user_clearance)
            resource_level_idx = CLEARANCE_LEVELS.index(resource_clearance)
            return user_level_idx >= resource_level_idx
        except ValueError:
            # If an unknown clearance string is passed, default to safe exclusion (unauthorized)
            return False

    def verify_access(self, username: str, resource_clearance: str) -> dict:
        """
        Performs access verification and returns a structured, explainable result.
        """
        user_clearance = self.get_user_clearance(username)
        authorized = self.is_authorized(user_clearance, resource_clearance)
        
        status = "AUTHORIZED" if authorized else "DENIED"
        explanation = (
            f"Access {status}: User '{username}' holds '{user_clearance}' clearance, "
            f"which is {'sufficient' if authorized else 'insufficient'} to access "
            f"'{resource_clearance}' classification documents."
        )

        return {
            "username": username,
            "user_clearance": user_clearance,
            "resource_clearance": resource_clearance,
            "authorized": authorized,
            "explanation": explanation
        }
