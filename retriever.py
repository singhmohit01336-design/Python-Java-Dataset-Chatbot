"""Strict Dataset-Only Retriever and Answer Engine for Python & Java Chatbot.

This module provides deterministic local retrieval over the verified Python and Java
Q&A dataset without external LLM dependencies, API keys, or network calls.
"""

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

REFUSAL_MESSAGE = "I can answer only questions about Python and Java programming covered by my dataset."

COMMON_QUESTION_WORDS = {
    'what', 'is', 'a', 'an', 'the', 'in', 'of', 'for', 'to', 'how', 'do',
    'you', 'does', 'can', 'explain', 'tell', 'me', 'about', 'describe',
    'difference', 'between', 'used', 'purpose', 'mean', 'work', 'why',
    'which', 'where', 'when', 'who', 'simple', 'words', 'please', 'give',
    'example', 'define', 'definition', 'works'
}

PROTECTED_CODE_TERMS = {
    'class', 'pass', 'this', 'super', 'final', 'yield', 'args', 'kwargs',
    'lambda', 'def', 'import', 'from', 'with', 'try', 'except', 'finally',
    'raise', 'return', 'async', 'await', 'static', 'public', 'private',
    'protected', 'void', 'int', 'float', 'double', 'boolean', 'new',
    'throw', 'throws', 'collection', 'collections'
}


def simple_stem(word: str) -> str:
    """Lightweight deterministic stemmer for English programming terms."""
    w = word.lower()
    if len(w) <= 3 or w in PROTECTED_CODE_TERMS:
        return w
    if w.endswith('sses'):
        return w[:-2]
    if w.endswith('ies'):
        return w[:-3] + 'y'
    if w.endswith('ing') and len(w) > 5:
        base = w[:-3]
        if base.endswith(('dd', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt')):
            base = base[:-1]
        return base
    if w.endswith('tion') or w.endswith('sion'):
        return w[:-4]
    if w.endswith('tions') or w.endswith('sions'):
        return w[:-5]
    if w.endswith('ed') and len(w) > 4:
        base = w[:-2]
        if base.endswith(('dd', 'gg', 'mm', 'nn', 'pp', 'rr', 'tt')):
            base = base[:-1]
        return base
    if w.endswith('s') and not w.endswith(('ss', 'us', 'is')):
        return w[:-1]
    return w


def clean_tokens(text: str, stem: bool = False) -> List[str]:
    """Tokenizes and normalizes text, preserving programming symbols like __name__."""
    if not text:
        return []
    # Replace non-alphanumeric (except underscores) with spaces
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned.split()
    if stem:
        words = [simple_stem(w) for w in words]
    return words


def get_content_words(words: List[str]) -> List[str]:
    """Filters out common non-discriminative question words."""
    return [w for w in words if w not in COMMON_QUESTION_WORDS and len(w) > 1]


class StrictDatasetRetriever:
    """Deterministic local retriever grounded strictly in the Python & Java dataset."""

    def __init__(self, dataset_path: Optional[str] = None):
        if dataset_path is None:
            # Default to data/python_java_chatbot_dataset_500.json relative to repo
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(base_dir, "data", "python_java_chatbot_dataset_500.json")

        self.dataset_path = dataset_path
        self.dataset: List[Dict[str, Any]] = []
        self.entries: List[Dict[str, Any]] = []
        self.all_content_stems = set()
        self.idf: Dict[str, float] = {}
        self.load_error: Optional[str] = None

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Loads and indexes the dataset safely."""
        if not os.path.exists(self.dataset_path):
            self.load_error = f"Dataset file not found at: {self.dataset_path}"
            return

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        except Exception as e:
            self.load_error = f"Failed to parse dataset JSON: {e}"
            return

        if not isinstance(self.dataset, list) or len(self.dataset) == 0:
            self.load_error = "Dataset JSON is empty or not a list of records."
            return

        doc_count = len(self.dataset)
        df: Counter = Counter()

        for idx, item in enumerate(self.dataset):
            q_words = clean_tokens(item.get("question", ""), stem=False)
            q_stems = [simple_stem(w) for w in q_words]
            q_content_stems = get_content_words(q_stems)
            self.all_content_stems.update(q_content_stems)

            a_words = clean_tokens(item.get("answer", ""), stem=False)
            a_stems = [simple_stem(w) for w in a_words]
            a_content_stems = get_content_words(a_stems)

            seen = set(q_stems + a_stems)
            for t in seen:
                df[t] += 1

            self.entries.append({
                "index": idx,
                "id": item.get("id", f"REC_{idx}"),
                "language": item.get("language", "General"),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "q_words": q_words,
                "q_stems": q_stems,
                "q_content_stems": set(q_content_stems),
                "a_content_stems": set(a_content_stems),
                "is_simple": "in simple words" in item.get("question", "").lower()
            })

        self.idf = {w: math.log((doc_count + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def retrieve(self, query: str, threshold: float = 0.40) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """Matches a user query strictly against the dataset entries."""
        if self.load_error:
            return None, 0.0, f"Error: {self.load_error}"

        if not query or not query.strip():
            return None, 0.0, "Empty query"

        raw_words = clean_tokens(query, stem=False)
        if not raw_words:
            return None, 0.0, "No valid tokens"

        q_stems = [simple_stem(w) for w in raw_words]
        q_content_stems = set(get_content_words(q_stems))

        has_python = 'python' in raw_words
        has_java = 'java' in raw_words and 'javascript' not in raw_words

        # Security & Scope Gate: If query contains content words, at least one MUST exist in dataset stems or language keywords
        if q_content_stems:
            matching_terms = q_content_stems.intersection(self.all_content_stems)
            if not matching_terms and not (has_python or has_java):
                return None, 0.0, "Out of domain keywords"

        query_is_simple = 'simple' in raw_words

        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        candidate_matches: List[Tuple[Dict[str, Any], float]] = []

        for entry in self.entries:
            lang_boost = 1.0
            if has_python and not has_java:
                if entry["language"] != "Python":
                    continue
                lang_boost = 1.25
            elif has_java and not has_python:
                if entry["language"] != "Java":
                    continue
                lang_boost = 1.25

            entry_q_content = entry["q_content_stems"]
            if not entry_q_content:
                continue

            # Question content overlap (F1)
            common_q_content = q_content_stems.intersection(entry_q_content)
            recall = len(common_q_content) / len(entry_q_content) if entry_q_content else 0.0
            precision = len(common_q_content) / len(q_content_stems) if q_content_stems else 0.0
            f1_content = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Answer content overlap
            common_a_content = q_content_stems.intersection(entry["a_content_stems"])
            a_overlap = len(common_a_content) / len(q_content_stems) if q_content_stems else 0.0

            # IDF weighted overlap
            shared_stems = set(q_stems).intersection(set(entry["q_stems"]))
            weighted_overlap = sum(self.idf.get(w, 1.0) for w in shared_stems)
            denom = sum(self.idf.get(w, 1.0) for w in entry["q_stems"]) + sum(self.idf.get(w, 1.0) for w in q_stems)
            idf_sim = (2 * weighted_overlap) / denom if denom > 0 else 0.0

            score = (0.50 * f1_content + 0.35 * idf_sim + 0.15 * a_overlap) * lang_boost

            # Exact question match bonus
            if raw_words == entry["q_words"]:
                score += 1.0

            # Penalize/boost 'in simple words' alignment
            if not query_is_simple and entry["is_simple"]:
                score *= 0.92
            elif query_is_simple and entry["is_simple"]:
                score *= 1.08

            if score >= threshold:
                candidate_matches.append((entry, score))

            if score > best_score:
                best_score = score
                best_entry = entry

        if not best_entry or best_score < threshold:
            return None, best_score, "Below confidence threshold"

        # Check for cross-language ambiguity (e.g. 'What is inheritance?') when no language is specified
        if not has_python and not has_java:
            # Check if there is an equally strong match in the other language
            other_lang = "Java" if best_entry["language"] == "Python" else "Python"
            other_candidates = [
                (e, s) for e, s in candidate_matches
                if e["language"] == other_lang
                and (abs(s - best_score) < 0.25 or e["q_words"] == best_entry["q_words"])
                and e["q_content_stems"] == best_entry["q_content_stems"]
            ]
            if other_candidates:
                other_entry = other_candidates[0][0]
                py_entry = best_entry if best_entry["language"] == "Python" else other_entry
                java_entry = best_entry if best_entry["language"] == "Java" else other_entry
                # Return synthetic combined representation grounded entirely in dataset answers
                combined = {
                    "id": f"{py_entry['id']} / {java_entry['id']}",
                    "language": "Python & Java",
                    "question": best_entry["question"],
                    "answer": f"**Python ({py_entry['id']}):** {py_entry['answer']}\n\n**Java ({java_entry['id']}):** {java_entry['answer']}",
                    "is_ambiguous": True
                }
                return combined, best_score, "Ambiguous cross-language match"

        return best_entry, best_score, "Matched"

    def answer_query(self, query: str, threshold: float = 0.40) -> str:
        """Returns the grounded dataset answer or strict refusal."""
        if self.load_error:
            return f"Service unavailable: {self.load_error}"

        entry, score, reason = self.retrieve(query, threshold=threshold)
        if entry:
            return entry["answer"]
        return REFUSAL_MESSAGE
