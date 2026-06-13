import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from rag.config import EMBEDDING_MODEL_NAME, SEMANTIC_SPLIT_PERCENTILE, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS

class SemanticChunker:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        split_percentile: float = SEMANTIC_SPLIT_PERCENTILE,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        max_chunk_chars: int = MAX_CHUNK_CHARS
    ):
        """
        Initializes the Semantic Chunker using a local SentenceTransformer.
        """
        self.model = SentenceTransformer(model_name)
        self.split_percentile = split_percentile
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using a robust splitting and stitching routine to preserve legal abbreviations.
        """
        # Clean white spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on sentence-ending punctuation followed by space (fixed-width lookbehind)
        raw_parts = re.split(r'(?<=[.!?])\s+', text)
        
        abbreviations = {
            "art.", "sect.", "no.", "vs.", "al.", "e.g.", "i.e.",
            "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."
        }
        
        sentences = []
        current_sentence = []
        
        for part in raw_parts:
            current_sentence.append(part)
            words = part.split()
            if words:
                last_word = words[-1].lower()
                # If the last word is a known abbreviation (or a single letter abbreviation like "A."), keep stitching
                if last_word in abbreviations or (len(last_word) == 2 and last_word.endswith('.')):
                    continue
            
            sentences.append(" ".join(current_sentence))
            current_sentence = []
            
        if current_sentence:
            sentences.append(" ".join(current_sentence))
            
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunks a document semantically by grouping sentences with high cosine similarity.
        Returns a list of dictionaries containing:
          - "text": The chunk text.
          - "start_char": The start character position in the original text.
          - "end_char": The end character position in the original text.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # Find character positions of sentences in original text to preserve metadata
        sentence_metadata = []
        current_pos = 0
        for sent in sentences:
            start_idx = text.find(sent, current_pos)
            if start_idx == -1:
                # Fallback if text cleaning caused mismatch
                start_idx = current_pos
            end_idx = start_idx + len(sent)
            sentence_metadata.append({
                "text": sent,
                "start": start_idx,
                "end": end_idx
            })
            current_pos = end_idx

        # If there is only one sentence or very short text, return it as a single chunk
        if len(sentences) < 2:
            return [{
                "text": text,
                "start_char": 0,
                "end_char": len(text)
            }]

        # 1. Embed sentences
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # 2. Compute cosine distances between consecutive sentences
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Prevent division by zero
        normalized_embeddings = embeddings / norms
        
        # Calculate dot product of consecutive vectors
        similarities = np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)
        distances = 1.0 - similarities

        # 3. Determine the splitting threshold based on distance percentile
        threshold = np.percentile(distances, self.split_percentile)

        # 4. Group sentences into chunks based on threshold and boundaries
        chunks = []
        current_chunk_sentences = [sentence_metadata[0]]
        current_chunk_text = sentence_metadata[0]["text"]

        for i in range(1, len(sentence_metadata)):
            distance = distances[i - 1]
            next_sentence = sentence_metadata[i]
            
            # Check length constraints
            too_long = (len(current_chunk_text) + len(next_sentence["text"]) + 1) > self.max_chunk_chars
            too_short = len(current_chunk_text) < self.min_chunk_chars
            
            # Decide whether to split
            # Split if distance exceeds threshold and chunk is not too short, or if adding would make it too long
            if (distance > threshold and not too_short) or too_long:
                # Save the current chunk
                chunks.append({
                    "text": current_chunk_text,
                    "start_char": current_chunk_sentences[0]["start"],
                    "end_char": current_chunk_sentences[-1]["end"]
                })
                # Start new chunk
                current_chunk_sentences = [next_sentence]
                current_chunk_text = next_sentence["text"]
            else:
                # Merge into current chunk
                current_chunk_sentences.append(next_sentence)
                current_chunk_text += " " + next_sentence["text"]

        # Append last remaining chunk
        if current_chunk_sentences:
            chunks.append({
                "text": current_chunk_text,
                "start_char": current_chunk_sentences[0]["start"],
                "end_char": current_chunk_sentences[-1]["end"]
            })

        return chunks
