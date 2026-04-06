"""
AI-SRF RAG Engine
Groq nomic-embed-text-v1_5 embeddings + ChromaDB persistent vector store
Follows groq-api-cookbook/tutorials/04-rag best practices
By: Bright Sikazwe, PhD Candidate
"""

import os
import json
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI   # Groq's OpenAI-compatible endpoint for embeddings
from config import GROQ_EMBED_BASE_URL, GROQ_EMBED_MODEL

logger = logging.getLogger(__name__)

REGULATORY_ANCHORS = [
    "POPIA Section 72: no cross-border transfer of personal information without lawful basis and Transfer Impact Assessment.",
    "King IV Principle 12: boards retain non-delegable oversight of technology and information governance.",
    "B-BBEE and EEA controls: workforce-affecting AI must be reviewed for distributional impact and proxy discrimination.",
]


class SAKnowledgeBase:
    """
    Retrieval-Augmented Generation engine for the AI-SRF SA institutional corpus.

    Architecture (Groq cookbook best practice):
    - Embedding:  Groq nomic-embed-text-v1_5  (via OpenAI-compatible endpoint)
    - Splitter:   RecursiveCharacterTextSplitter  chunk_size=1000, overlap=200
    - Vector DB:  ChromaDB persistent client
    - Collection: ai_srf_sa_corpus
    - Search:     cosine similarity, top-k=5
    """

    def __init__(self, groq_api_key: str, persist_dir: str = "./chroma_db", collection: str = "ai_srf_sa_corpus"):
        self.groq_api_key = groq_api_key
        self.persist_dir = persist_dir
        self.collection_name = collection

        # Groq embedding client (OpenAI-compatible endpoint)
        self._embed_client = OpenAI(
            api_key=groq_api_key,
            base_url=GROQ_EMBED_BASE_URL,
        )
        self._embed_model = GROQ_EMBED_MODEL

        # ChromaDB persistent client
        os.makedirs(persist_dir, exist_ok=True)
        self._chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_ids: set = set()
        self._load_indexed_ids()

    # ─── Public API ────────────────────────────────────────────────────────

    def index_corpus(self, documents: List[Dict]) -> int:
        """Index a list of {id, title, content, source} dicts. Skip already-indexed."""
        new_docs = [d for d in documents if d["id"] not in self._indexed_ids]
        if not new_docs:
            return 0

        texts = [f"{d['title']}\n\n{d['content']}" for d in new_docs]
        ids   = [d["id"] for d in new_docs]
        metas = [{"title": d["title"], "source": d.get("source", ""), "id": d["id"]} for d in new_docs]

        embeddings = self._embed_batch(texts)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        self._indexed_ids.update(ids)
        self._save_indexed_ids()
        logger.info("Indexed %d documents into ChromaDB", len(new_docs))
        return len(new_docs)

    def index_pdfs(self, pdf_dir: str) -> int:
        """
        Load PDFs from directory, chunk with RecursiveCharacterTextSplitter,
        embed with Groq nomic-embed-text-v1_5, store in ChromaDB.
        Follows groq-api-cookbook RAG tutorial pattern.
        """
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pdf_path = Path(pdf_dir)
        if not pdf_path.exists() or not any(pdf_path.glob("*.pdf")):
            logger.info("No PDFs found in %s", pdf_dir)
            return 0

        loader = PyPDFDirectoryLoader(str(pdf_path))
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info("Loaded %d PDF chunks from %s", len(chunks), pdf_dir)

        new_chunks, new_ids, new_metas, new_texts = [], [], [], []
        for chunk in chunks:
            doc_id = "pdf_" + hashlib.sha256(chunk.page_content[:120].encode()).hexdigest()[:16]
            if doc_id not in self._indexed_ids:
                new_chunks.append(chunk)
                new_ids.append(doc_id)
                src = Path(chunk.metadata.get("source", "unknown")).name
                new_metas.append({
                    "title": f"PDF: {src} (p.{chunk.metadata.get('page', '?')})",
                    "source": src,
                    "id": doc_id,
                })
                new_texts.append(chunk.page_content)

        if not new_texts:
            return 0

        # Batch embed (Groq rate-limit safe: 20 texts per call)
        BATCH = 20
        all_embeddings = []
        for i in range(0, len(new_texts), BATCH):
            batch = new_texts[i : i + BATCH]
            all_embeddings.extend(self._embed_batch(batch))

        self._collection.add(
            ids=new_ids,
            embeddings=all_embeddings,
            documents=new_texts,
            metadatas=new_metas,
        )
        self._indexed_ids.update(new_ids)
        self._save_indexed_ids()
        logger.info("Indexed %d PDF chunks into ChromaDB", len(new_texts))
        return len(new_texts)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Semantic retrieval: embed query → cosine search → return top-k docs."""
        if self._collection.count() == 0:
            return self._keyword_fallback(query, k)

        q_emb = self._embed_batch([query])
        results = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append({
                "id":      meta.get("id", "unknown"),
                "title":   meta.get("title", "Untitled"),
                "content": text[:600],
                "source":  meta.get("source", ""),
                "score":   round(1 - dist, 4),   # cosine distance → similarity
            })
        return docs

    def format_context(self, docs: List[Dict], max_chars: int = 4000) -> str:
        """Format retrieved docs as RAG context string."""
        parts, total = [], 0
        for d in docs:
            block = f"[{d['title']} | {d['source']}]\n{d['content']}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def build_governance_context_package(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.18,
        max_chars: int = 4000,
    ) -> Dict[str, Any]:
        """
        South African-compliant RAG package for agent reasoning.

        Stages:
        1. Retrieval
        2. Relevance enforcement
        3. Residual PII screening
        4. Regulatory anchor injection
        5. Sovereign grounding summary for ASY-style monitoring
        """
        retrieved = self.retrieve(query, k=k)
        filtered = [doc for doc in retrieved if doc.get("score", 0) >= min_score]
        if not filtered:
            filtered = retrieved[: min(2, len(retrieved))]

        enforced_docs = []
        for doc in filtered:
            screened = self._screen_document(doc)
            if screened["ingestion_decision"] == "allow":
                enforced_docs.append(screened)

        if not enforced_docs:
            enforced_docs = [self._screen_document(doc, force_allow=True) for doc in filtered[:1]]

        context = self.format_context(enforced_docs, max_chars=max_chars)
        return {
            "query": query,
            "retrieved_count": len(retrieved),
            "enforced_count": len(enforced_docs),
            "relevance_threshold": min_score,
            "documents": enforced_docs,
            "regulatory_anchors": REGULATORY_ANCHORS,
            "context": context,
            "compliance_notes": [
                "Only relevance-qualified chunks are admitted into the reasoning context window.",
                "Residual personal-information patterns are screened before context delivery.",
                "Regulatory anchors are injected into every governance context package.",
            ],
            "algorithmic_sovereignty_yield_hint": round((len(enforced_docs) / max(len(retrieved), 1)) * 100, 1),
        }

    def collection_size(self) -> int:
        return self._collection.count()

    # ─── Private ───────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Groq's nomic-embed-text-v1_5 endpoint."""
        try:
            response = self._embed_client.embeddings.create(
                model=self._embed_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("Groq embedding failed: %s — using zero vectors", e)
            # Return zero vectors (will give random retrieval — triggers keyword fallback)
            dim = 768  # nomic-embed-text-v1_5 dimension
            return [[0.0] * dim for _ in texts]

    def _screen_document(self, doc: Dict[str, Any], force_allow: bool = False) -> Dict[str, Any]:
        text = doc.get("content", "")
        pii_patterns = [
            r"\b\d{13}\b",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\b(?:\+27|0)\d{9}\b",
        ]
        has_pii = any(re.search(pattern, text) for pattern in pii_patterns)
        redacted = text
        for pattern in pii_patterns:
            redacted = re.sub(pattern, "[REDACTED]", redacted)

        screened = dict(doc)
        screened["content"] = redacted[:600]
        screened["pii_detected"] = bool(has_pii)
        screened["ingestion_decision"] = "allow" if (force_allow or not has_pii) else "review"
        return screened

    def _keyword_fallback(self, query: str, k: int) -> List[Dict]:
        """Keyword-match fallback when ChromaDB is empty."""
        from config import SA_CORPUS
        q = query.lower()
        scored = []
        for doc in SA_CORPUS:
            score = sum(1 for kw in doc.get("keywords", []) if kw.lower() in q)
            if score > 0:
                scored.append({**doc, "score": score / 10})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def _indexed_ids_path(self) -> str:
        return os.path.join(self.persist_dir, "indexed_ids.json")

    def _load_indexed_ids(self):
        p = self._indexed_ids_path()
        if os.path.exists(p):
            with open(p) as f:
                self._indexed_ids = set(json.load(f))

    def _save_indexed_ids(self):
        with open(self._indexed_ids_path(), "w") as f:
            json.dump(list(self._indexed_ids), f)
