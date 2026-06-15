# Orbyte AI - Legal RAG System Test Scenarios

This document outlines the test cases and expected outcomes to verify the Multi-Agent Legal RAG System. The tests cover user identity verification, security/clearance enforcement, factual grounding, and judge feedback across all clearance levels.

---

## Clearance Reference Table

| User | Clearance Level | Accessible Documents | Restricted Documents |
| :--- | :--- | :--- | :--- |
| **Charlie** | `Public` | Public only | Internal, Confidential |
| **Bob** | `Internal` | Public, Internal | Confidential |
| **Alice** | `Confidential` | Public, Internal, Confidential | None (All) |

---

## Test Cases

### 1. Public User (`charlie`) - Authorized Access
*   **Query**: `What are the essential components in managing contractual disputes?`
*   **Target Document**: `contractual dispute management` (clearance: `Public`)
*   **Expected Results**:
    *   **Retrieval**: The system finds the document chunk.
    *   **Access Check**: Authorized (Public query on Public document).
    *   **Response**: Orbyte AI answers that the essential components are **proper documentation and clear communication**.
    *   **Access Notice**: **No** security or clearance notices should be printed.
    *   **Judge Evaluation**: Faithfulness: `5/5`, Approved: `True`.

### 2. Public User (`charlie`) - Unauthorized Access
*   **Query**: `What does a robust regulatory policy compliance strategy involve?`
*   **Target Document**: `regulatory policy compliance strategy` (clearance: `Confidential`)
*   **Expected Results**:
    *   **Retrieval**: The document matches the query but is filtered out due to clearance.
    *   **Access Check**: Denied.
    *   **Response**: The system must explicitly output:
        ```text
        Notice: Some matching documents were filtered out due to insufficient access clearance level. I cannot answer this query based on the provided documents.
        ```
    *   **Judge Evaluation**: Faithfulness: `5/5` (compliant refusal), Approved: `True`.

---

### 3. Internal User (`bob`) - Authorized Access
*   **Query**: `How does optimizing the legal document authentication process improve efficiency?`
*   **Target Document**: `legal document authentication process optimization` (clearance: `Internal`)
*   **Expected Results**:
    *   **Retrieval**: The system retrieves the document chunk.
    *   **Access Check**: Authorized (Internal user querying Internal document).
    *   **Response**: Answers that it involves the implementation of advanced technologies such as **blockchain and AI** to enhance security, reduce fraud, and streamline the verification process.
    *   **Access Notice**: **No** security warnings are output.
    *   **Judge Evaluation**: Faithfulness: `5/5`, Approved: `True`.

### 4. Internal User (`bob`) - Unauthorized Access
*   **Query**: `What happens if a force-majeure condition endures for one and a half months?`
*   **Target Document**: `ambiguous force-majeure termination clause` (clearance: `Confidential`)
*   **Expected Results**:
    *   **Retrieval**: Matches the ambiguous force-majeure clause but filters it.
    *   **Access Check**: Denied.
    *   **Response**: The system must explicitly output:
        ```text
        Notice: Some matching documents were filtered out due to insufficient access clearance level. I cannot answer this query based on the provided documents.
        ```
    *   **Judge Evaluation**: Faithfulness: `5/5` (compliant refusal), Approved: `True`.

---

### 5. Confidential User (`alice`) - Authorized Access (Confidential Document)
*   **Query**: `What happens if a force-majeure condition endures for one and a half months?`
*   **Target Document**: `ambiguous force-majeure termination clause` (clearance: `Confidential`)
*   **Expected Results**:
    *   **Retrieval**: The system retrieves the force-majeure clause document chunk.
    *   **Access Check**: Authorized (Confidential user querying Confidential document).
    *   **Response**: Answers that the client may elect to terminate the agreement without incurring penalties, but remains bound by the support obligations (timely hand-over of services, data, and related documentation).
    *   **Access Notice**: **No** security warnings are output.
    *   **Judge Evaluation**: Faithfulness: `5/5`, Approved: `True`.

---

### 6. Hallucination Check - Out-of-Context Query (Any User)
*   **Query**: `What is the penalty fee for failing the cybersecurity compliance training?`
*   **Target Context**: `regulatory policy compliance strategy` mentions compliance training, but does not list any fees.
*   **Expected Results**:
    *   **Response**: The system should refuse to make up a penalty fee or external details. It should respond stating that it cannot answer based on the provided documents (or security refusal if unauthorized).
    *   **Judge Evaluation**: The Judge Agent checks that the LLM does not hallucinate numbers or facts.
