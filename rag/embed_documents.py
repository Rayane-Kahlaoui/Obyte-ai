import argparse
import sys
import os
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

# Add parent directory to path to enable local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.config import (
    DATASET_NAME,
    DATABASE_DIR,
    DATA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CLEARANCE_LEVELS
)
from rag.semantic_chunker import SemanticChunker

def main():
    parser = argparse.ArgumentParser(description="Embed legal documents into local ChromaDB")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of documents to embed (for testing)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for database insertion")
    args = parser.parse_args()

    print("Initializing Semantic Chunker...")
    # This will load the sentence transformer model locally
    chunker = SemanticChunker()

    parquet_path = os.path.join(DATA_DIR, "train-00000-of-00001.parquet")
    if os.path.exists(parquet_path):
        print(f"Loading local Parquet dataset from '{parquet_path}'...")
        try:
            dataset = load_dataset("parquet", data_files=parquet_path)
        except Exception as e:
            print(f"\nERROR: Could not load local Parquet dataset: {e}")
            raise e
    else:
        print(f"Loading Hugging Face dataset '{DATASET_NAME}'...")
        try:
            dataset = load_dataset(DATASET_NAME)
        except Exception as e:
            print("\n" + "="*80)
            print("ERROR: Could not load the dataset.")
            print("This dataset is gated. Please verify that:")
            print("1. You have logged in to Hugging Face using: huggingface-cli login")
            print("2. Or set the HF_TOKEN environment variable in your terminal.")
            print("="*80 + "\n")
            raise e

    # We assume 'train' split contains the document text
    documents = dataset.get("train")
    if not documents:
        # Fallback to whatever split is available
        split_name = list(dataset.keys())[0]
        documents = dataset[split_name]
        print(f"No 'train' split found. Using '{split_name}' split.")

    num_docs = len(documents)
    limit = args.limit if args.limit is not None else num_docs
    limit = min(limit, num_docs)
    
    print(f"Found {num_docs} documents. Ingesting {limit} documents...")

    # Initialize local ChromaDB client
    print(f"Connecting to persistent ChromaDB at '{DATABASE_DIR}'...")
    chroma_client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    
    # Initialize embedding function
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    # Ingestion buffers
    all_ids = []
    all_documents = []
    all_metadatas = []

    print("Processing and chunking documents semantically...")
    for doc_idx in tqdm(range(limit)):
        doc = documents[doc_idx]
        
        # The text column is usually 'text' or 'content'
        doc_text = doc.get("text", "")
        if not doc_text:
            continue

        # Generate semantic chunks
        chunks = chunker.chunk_document(doc_text)
        
        # Round-robin assign security levels (Public, Internal, Confidential)
        # to test our Identity/Permission Agent
        clearance = CLEARANCE_LEVELS[doc_idx % len(CLEARANCE_LEVELS)]

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            
            all_ids.append(chunk_id)
            all_documents.append(chunk["text"])
            all_metadatas.append({
                "doc_id": doc_idx,
                "chunk_idx": chunk_idx,
                "clearance": clearance,
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "source": f"HF_dataset_row_{doc_idx}"
            })

    total_chunks = len(all_ids)
    print(f"Generated {total_chunks} semantic chunks from {limit} documents.")
    print("Writing chunks to ChromaDB database...")

    # Add to ChromaDB in batches to prevent payload limits
    batch_size = args.batch_size
    for i in tqdm(range(0, total_chunks, batch_size)):
        end_idx = min(i + batch_size, total_chunks)
        collection.add(
            ids=all_ids[i:end_idx],
            documents=all_documents[i:end_idx],
            metadatas=all_metadatas[i:end_idx]
        )

    print(f"Successfully embedded and loaded {total_chunks} chunks into ChromaDB.")

if __name__ == "__main__":
    main()
