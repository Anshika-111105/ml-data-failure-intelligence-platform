"""
Runbook Vector Store & Semantic Retrieval Engine.
Indexes synthetic runbooks using Sentence-Transformers and FAISS (with TF-IDF cosine fallback)
to provide fast, evidence-grounded passage retrieval for incident investigation.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RUNBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "runbooks")


class RunbookChunk:
    def __init__(self, runbook_id: str, title: str, section: str, content: str, file_name: str):
        self.runbook_id = runbook_id
        self.title = title
        self.section = section
        self.content = content
        self.file_name = file_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runbook_id": self.runbook_id,
            "title": self.title,
            "section": self.section,
            "content": self.content,
            "file_name": self.file_name
        }


def parse_runbooks_to_chunks(runbooks_dir: str = RUNBOOKS_DIR) -> List[RunbookChunk]:
    """Parses markdown runbooks into structured semantic sections."""
    chunks = []
    if not os.path.exists(runbooks_dir):
        logger.warning(f"Runbooks directory not found: {runbooks_dir}")
        return chunks

    for file_name in sorted(os.listdir(runbooks_dir)):
        if not file_name.endswith(".md"):
            continue

        file_path = os.path.join(runbooks_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract Runbook ID and Title
        id_match = re.search(r"\*\*ID:\*\*\s*(RB-\d+)", text)
        runbook_id = id_match.group(1) if id_match else file_name.replace(".md", "")

        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1) if title_match else file_name

        # Split by level-2 sections (##)
        sections = re.split(r"\n##\s+", text)
        for idx, sec in enumerate(sections):
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            if idx == 0:
                # Header metadata section
                section_title = "Overview & Metadata"
                section_body = sec_clean
            else:
                lines = sec_clean.split("\n", 1)
                section_title = lines[0].strip()
                section_body = lines[1].strip() if len(lines) > 1 else ""

            if len(section_body) > 30:
                chunks.append(RunbookChunk(
                    runbook_id=runbook_id,
                    title=title,
                    section=section_title,
                    content=section_body,
                    file_name=file_name
                ))

    logger.info(f"Parsed {len(chunks)} semantic chunks from {runbooks_dir}")
    return chunks


class RunbookVectorStore:
    def __init__(self, runbooks_dir: str = RUNBOOKS_DIR, model_name: str = "all-MiniLM-L6-v2"):
        self.runbooks_dir = runbooks_dir
        self.model_name = model_name
        self.chunks: List[RunbookChunk] = []
        self.index = None
        self.embedding_model = None
        self.use_fallback = False
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self._initialize()

    def _initialize(self):
        """Loads runbooks and builds embedding index."""
        self.chunks = parse_runbooks_to_chunks(self.runbooks_dir)
        if not self.chunks:
            logger.warning("No runbook chunks available to index.")
            return

        texts = [f"{c.runbook_id} {c.title} - {c.section}: {c.content}" for c in self.chunks]

        # Attempt to load SentenceTransformers + FAISS
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            logger.info(f"Loading embedding model '{self.model_name}'...")
            self.embedding_model = SentenceTransformer(self.model_name)
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner Product for normalized vectors = Cosine Similarity
            self.index.add(embeddings.astype(np.float32))
            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.warning(f"Dense vector indexing failed or offline ({e}). Using robust TF-IDF Cosine fallback.")
            self.use_fallback = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant runbook passages matching the query."""
        if not self.chunks:
            return []

        results = []
        k = min(top_k, len(self.chunks))

        if not self.use_fallback and self.embedding_model is not None and self.index is not None:
            query_vec = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = self.index.search(query_vec.astype(np.float32), k)

            for score, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    res_dict = chunk.to_dict()
                    res_dict["similarity_score"] = float(np.round(score, 4))
                    results.append(res_dict)
        else:
            # Fallback TF-IDF cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(scores)[::-1][:k]

            for idx in top_indices:
                chunk = self.chunks[idx]
                res_dict = chunk.to_dict()
                res_dict["similarity_score"] = float(np.round(scores[idx], 4))
                results.append(res_dict)

        return results


if __name__ == "__main__":
    store = RunbookVectorStore()
    test_queries = [
        "high missing null rate in UnitPrice feature",
        "extreme 100x price scale jump and outlier ceiling",
        "unknown country category ATLANTIS_DEPOT",
        "quantity distribution drift high PSI bulk buyers"
    ]
    for q in test_queries:
        hits = store.search(q, top_k=2)
        print(f"\nQuery: '{q}'")
        for h in hits:
            print(f" -> [{h['runbook_id']}: {h['section']}] (Score: {h['similarity_score']}): {h['content'][:90]}...")
