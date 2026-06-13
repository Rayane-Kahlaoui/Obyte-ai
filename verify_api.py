import sys
import os

# Force mock mode for testing
os.environ["LLM_API_TYPE"] = "mock"

# Add parent directory to path to enable local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    print("Testing /api/v1/health...")
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "chunk_count" in data
    assert "llm_mode" in data
    print("[PASS] Health endpoint test passed.")

def test_auth_verify():
    print("Testing /api/v1/auth/verify...")
    # Bob (Internal) -> Confidential: Denied
    response = client.get("/api/v1/auth/verify?user=bob&document_clearance=Confidential")
    assert response.status_code == 200
    data = response.json()
    assert data["authorized"] is False
    assert "DENIED" in data["explanation"]

    # Alice (Confidential) -> Public: Authorized
    response = client.get("/api/v1/auth/verify?user=alice&document_clearance=Public")
    assert response.status_code == 200
    data = response.json()
    assert data["authorized"] is True
    assert "AUTHORIZED" in data["explanation"]
    print("[PASS] Auth verification tests passed.")

def test_audit():
    print("Testing /api/v1/audit...")
    payload = {
        "query": "What training is required?",
        "retrieved_contexts": ["The compliance policy requires annual cybersecurity training for all staff."],
        "response": "[Local Mock LLM Response - Strict Mode] The compliance policy requires annual cybersecurity training for all staff."
    }
    response = client.post("/api/v1/audit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["approved"] is True
    assert data["faithfulness_score"] == 5
    print("[PASS] Audit endpoint tests passed.")

def test_query_rag():
    print("Testing /api/v1/query...")
    # Test nominal case as Alice (Confidential)
    payload = {
        "username": "alice",
        "query": "compliance training requirements",
        "top_k": 2
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "alice"
    assert "response" in data
    assert "evaluation" in data
    assert len(data["chunks_used"]) <= 2

    # Test forbidden case (Charlie asking for confidential compliance info, assuming no public doc matches)
    # Note: Charlie (Public) asking about something that will query vector DB
    # Let's verify if query handles empty context or returns 403 when only unauthorized chunks exist.
    # To test this robustly, we ask about a strategic supplier contract (Confidential level in assign_clearance)
    payload_charlie = {
        "username": "charlie",
        "query": "Strategic supplier agreement details",
        "top_k": 2
    }
    # This might return 403 Forbidden if all matching chunks are Confidential/Internal
    response_charlie = client.post("/api/v1/query", json=payload_charlie)
    # The response can be 200 (if some public chunks are returned or if the retrieval matched nothing)
    # or 403 (if match attempts were all unauthorized). Let's print the status and assert valid responses
    print(f"Charlie query status: {response_charlie.status_code}")
    assert response_charlie.status_code in [200, 403]
    if response_charlie.status_code == 403:
        assert "Access denied" in response_charlie.json()["detail"]
        print("[PASS] Query endpoint correctly denied access (403 Forbidden).")
    else:
        print("[PASS] Query endpoint completed successfully (200 OK).")

if __name__ == "__main__":
    print("=========================================")
    print("Running Orbyte REST API Integration Tests")
    print("=========================================")
    try:
        test_health()
        test_auth_verify()
        test_audit()
        test_query_rag()
        print("=========================================")
        print("ALL API TESTS COMPLETED SUCCESSFULLY!")
        print("=========================================")
    except AssertionError as e:
        print("\n[FAIL] API TEST FAILURE:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED API ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
