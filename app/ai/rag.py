import os
import re
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings

logger = logging.getLogger(__name__)


def medical_tokenizer(text: str) -> List[str]:
    """Tokenize and apply lightweight morphological stemming to match inflections."""
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    stemmed = []
    for w in tokens:
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("es") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        elif w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        stemmed.append(w)
    return stemmed


class MedicalRAG:
    """
    Lightweight, fast RAG engine for medical document retrieval using TF-IDF.
    Provides context grounding from the configured medical Q&A CSV.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or settings.MEDQUAD_CSV_PATH
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.df: Optional[pd.DataFrame] = None
        self._initialize()

    def _initialize(self):
        """Load dataset and build TF-IDF search index."""
        target_path = self.data_path
        if not os.path.exists(target_path):
            # Fallback to built-in sample if configured path does not exist
            sample_path = "./data/sample_medquad.csv"
            if os.path.exists(sample_path):
                logger.info(f"Dataset '{target_path}' not found. Falling back to '{sample_path}'.")
                target_path = sample_path
            else:
                logger.warning(f"No medical dataset found at '{target_path}' or '{sample_path}'.")
                return

        try:
            df = pd.read_csv(target_path)
            # Normalize column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if 'question' not in df.columns or 'answer' not in df.columns:
                logger.error("Dataset must contain 'question' and 'answer' columns.")
                return

            self.df = df[['question', 'answer']].dropna().reset_index(drop=True)
            search_corpus = (self.df['question'] + " " + self.df['answer']).tolist()

            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                tokenizer=medical_tokenizer,
                token_pattern=None,
                stop_words='english'
            )
            self.matrix = self.vectorizer.fit_transform(search_corpus)
            logger.info(f"Medical RAG index ready with {len(self.df)} records.")

        except Exception as e:
            logger.error(f"Failed to initialize Medical RAG: {e}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve the top-k most relevant medical Q&A entries for the user query.
        Returns list of dicts with: question, answer, score.
        """
        if self.vectorizer is None or self.matrix is None or self.df is None or len(self.df) == 0:
            return []

        k = top_k or settings.RAG_TOP_K
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.05:  # Relevance threshold
                    results.append({
                        "question": str(self.df.iloc[idx]['question']),
                        "answer": str(self.df.iloc[idx]['answer']),
                        "score": round(score, 4)
                    })
            return results
        except Exception as e:
            logger.error(f"RAG retrieval error for query '{query}': {e}")
            return []

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict],
        user_profile: Optional[Dict] = None
    ) -> str:
        """Construct grounded prompt combining query and retrieved context."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}] Q: {doc['question']}\nA: {doc['answer']}")

        context_str = "\n\n".join(context_parts) if context_parts else "No specific reference documents found."

        tone_hint = "clear, empathetic, and layman-friendly"
        if user_profile:
            age = user_profile.get("age")
            if age and age >= 60:
                tone_hint = "formal, clear, and reassuring"

        prompt = (
            f"You are a trustworthy healthcare assistant.\n\n"
            f"Tone: {tone_hint}\n\n"
            f"Retrieved Medical Knowledge:\n"
            f"{context_str}\n\n"
            f"Patient Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Provide a concise, helpful 2-3 sentence answer directly answering the question.\n"
            f"2. Ground your response in the provided medical knowledge.\n"
            f"3. Do not invent drug dosages or unverified claims."
        )
        return prompt


# Singleton instance
rag_engine = MedicalRAG()
