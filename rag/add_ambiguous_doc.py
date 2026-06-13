import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import chromadb
from chromadb.utils import embedding_functions
from rag.semantic_chunker import SemanticChunker
from rag.config import (
    DATABASE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CLEARANCE_LEVELS,
)

# ----------------------------------------------------------------------
# Ambiguous contract clause (intentionally vague)
# ----------------------------------------------------------------------
ambiguous_text = """
The Service Agreement indicates that if a force‑majeure condition endures for a period
approximately equal to one and a half months, the client may elect to terminate the
agreement without incurring penalties. Upon such termination, the client remains bound
by the support obligations listed in an exhibit, which require a timely hand‑over of
services, data, and related documentation to the client or its successor.
"""

# ----------------------------------------------------------------------
# One‑row DataFrame matching the existing schema
# ----------------------------------------------------------------------
df = pd.DataFrame([{
    "keyword": "ambiguous force‑majeure termination clause",
    "topic":   "Legal Documents",
    "language":"English",
    "text":    ambiguous_text.strip()
}])

# ----------------------------------------------------------------------
# Write to a local Parquet file (same folder as the other data)
# ----------------------------------------------------------------------
data_dir = pathlib.Path(r"C:/Users/rayan/Desktop/Orbyte/data")
data_dir.mkdir(parents=True, exist_ok=True)
parquet_path = data_dir / "ambiguous_contract.parquet"
table = pa.Table.from_pandas(df)
pq.write_table(table, parquet_path)
print(f"[OK] Ambiguous contract written to {parquet_path}")

# ----------------------------------------------------------------------
# Ingest the document into ChromaDB (reuse the existing pipeline)
# ----------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn,
)

# Chunk the text
chunker = SemanticChunker()
chunks = chunker.chunk_document(ambiguous_text.strip())

# Assign a high clearance so the IdentityAgent will allow the query
clearance = CLEARANCE_LEVELS[-1]   # "Confidential"

ids, docs, metas = [], [], []
for i, chunk in enumerate(chunks):
    chunk_id = f"ambiguous_contract_chunk_{i}"
    ids.append(chunk_id)
    docs.append(chunk["text"])
    metas.append({
        "doc_id":    "ambiguous_contract",
        "chunk_idx": i,
        "clearance": clearance,
        "start_char": chunk["start_char"],
        "end_char":   chunk["end_char"],
        "source":     "ambiguous_contract.parquet"
    })

collection.add(ids=ids, documents=docs, metadatas=metas)
print(f"✅ Ingested {len(ids)} semantic chunk(s) into ChromaDB")
