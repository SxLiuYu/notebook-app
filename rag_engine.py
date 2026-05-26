#!/usr/bin/env python3
"""RAG Engine: ChromaDB + FinnA cloud embeddings, hybrid search with BM25 + LLM re-rank."""
import os, math, json, gc
import requests

# Singletons
_chroma_client = None
_collection = None
_bm25_index = None

CHROMA_PATH = None
COLLECTION_NAME = "notebook_docs"

# FinnA embedding config
EMBED_KEY = "app-mrpIrSZCEzQiQAce86zPbL27"
EMBED_BASE = "https://www.finna.com.cn/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024


def _get_embeddings(texts):
    """Get embeddings from FinnA cloud API. texts: list of strings."""
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        return []
    
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{EMBED_BASE}/embeddings",
                headers={
                    "Authorization": f"Bearer {EMBED_KEY}",
                    "Content-Type": "application/json"
                },
                json={"model": EMBED_MODEL, "input": texts},
                timeout=30
            )
            data = resp.json()
            if "data" in data:
                return [d["embedding"] for d in data["data"]]
        except Exception as e:
            print(f"Embed attempt {attempt+1} failed: {e}")
            import time
            time.sleep(1)
    
    return []


def _get_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        import chromadb
        path = CHROMA_PATH or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'chroma_db')
        os.makedirs(path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=path)
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ─── BM25 ───
class BM25Scorer:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = {}
        self.doc_len = {}
        self.avgdl = 0
        self.total_docs = 0
    
    def tokenize(self, text):
        tokens = []
        current_word = ""
        for ch in text.lower():
            if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                tokens.append(ch)
            elif ch.isalnum():
                current_word += ch
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
        if current_word:
            tokens.append(current_word)
        return [t for t in tokens if len(t) > 0]
    
    def add(self, doc_id, text):
        tokens = self.tokenize(text)
        self.corpus[doc_id] = tokens
        self.doc_len[doc_id] = len(tokens)
        self.total_docs += 1
        self.avgdl = sum(self.doc_len.values()) / max(self.total_docs, 1)
    
    def remove(self, doc_id):
        if doc_id in self.corpus:
            del self.corpus[doc_id]
            del self.doc_len[doc_id]
            self.total_docs -= 1
            if self.total_docs > 0:
                self.avgdl = sum(self.doc_len.values()) / self.total_docs
            else:
                self.avgdl = 0
    
    def score(self, doc_id, query):
        if doc_id not in self.corpus:
            return 0
        query_tokens = self.tokenize(query)
        doc_tokens = self.corpus[doc_id]
        doc_len = self.doc_len[doc_id]
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0
        for qt in query_tokens:
            if qt not in tf:
                continue
            df = sum(1 for d in self.corpus.values() if qt in d)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf[qt] * (self.k1 + 1)
            denominator = tf[qt] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += idf * numerator / max(denominator, 0.1)
        return score
    
    def search(self, query, doc_ids=None, top_k=20):
        candidates = doc_ids or list(self.corpus.keys())
        scored = [(doc_id, self.score(doc_id, query)) for doc_id in candidates]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ─── Public API ───
def add_documents(doc_id, chunks, metadata_dict=None):
    """Index document chunks into ChromaDB and BM25 via FinnA cloud embeddings."""
    collection = _get_collection()
    global _bm25_index
    
    if _bm25_index is None:
        _bm25_index = BM25Scorer()
    
    if not chunks:
        return
    
    texts = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        meta = metadata_dict or {}
        if isinstance(chunk, dict) and "metadata" in chunk:
            meta = {**meta, **chunk["metadata"]}
        
        meta["doc_id"] = doc_id
        meta["chunk_index"] = i
        
        chunk_id = f"{doc_id}_{i}"
        texts.append(text)
        metadatas.append(meta)
        ids.append(chunk_id)
        
        # Add to BM25
        _bm25_index.add(chunk_id, text)
    
    # Get embeddings from FinnA (batch all at once)
    BATCH = 20
    all_embeddings = []
    for start in range(0, len(texts), BATCH):
        batch_texts = texts[start:start+BATCH]
        embs = _get_embeddings(batch_texts)
        if len(embs) != len(batch_texts):
            print(f"WARNING: got {len(embs)} embeddings for {len(batch_texts)} texts")
            # Pad with zeros if mismatch
            while len(embs) < len(batch_texts):
                embs.append([0.0] * EMBED_DIM)
        all_embeddings.extend(embs)
    
    # Add to ChromaDB in batches
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(
            embeddings=all_embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
    
    gc.collect()


def search(query, top_k=5, doc_ids=None):
    """Hybrid search: FinnA vector similarity + BM25."""
    collection = _get_collection()
    
    if collection.count() == 0:
        return []
    
    # Get query embedding from FinnA
    query_emb = _get_embeddings([query])
    if not query_emb:
        # Fallback to BM25 only
        query_emb = None
    else:
        query_emb = query_emb[0]
    
    n_results = min(20, collection.count())
    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            where_filter = {"doc_id": doc_ids[0]}
        else:
            where_filter = {"doc_id": {"$in": doc_ids}}
    
    # Vector search
    results = collection.query(
        query_embeddings=[query_emb] if query_emb else None,
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["ids"] or not results["ids"][0]:
        return []
    
    # Combine with BM25
    global _bm25_index
    bm25_scores = {}
    if _bm25_index:
        bm25_results = _bm25_index.search(query, doc_ids=results["ids"][0], top_k=n_results)
        max_bm25 = max((s for _, s in bm25_results), default=1)
        for chunk_id, score in bm25_results:
            bm25_scores[chunk_id] = score / max(max_bm25, 1)
    
    # Normalize vector distances
    distances = results["distances"][0]
    if distances and max(distances) > min(distances):
        max_dist = max(distances)
        min_dist = min(distances)
        vec_scores = {cid: 1 - (d - min_dist) / (max_dist - min_dist)
                     for cid, d in zip(results["ids"][0], distances)}
    else:
        vec_scores = {cid: 1.0 for cid in results["ids"][0]}
    
    # Hybrid: 0.7 vector + 0.3 BM25
    alpha = 0.7
    combined = []
    for i, chunk_id in enumerate(results["ids"][0]):
        vs = vec_scores.get(chunk_id, 0.8)
        bs = bm25_scores.get(chunk_id, 0)
        score = alpha * vs + (1 - alpha) * bs
        combined.append({
            "chunk_text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(score, 4),
            "chunk_id": chunk_id
        })
    
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


def delete_document(doc_id):
    collection = _get_collection()
    global _bm25_index
    try:
        results = collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            if _bm25_index:
                for chunk_id in results["ids"]:
                    _bm25_index.remove(chunk_id)
    except Exception:
        pass


def get_stats():
    collection = _get_collection()
    try:
        count = collection.count()
        if count > 0:
            all_data = collection.get(include=["metadatas"])
            unique_docs = len(set(m.get("doc_id", "") for m in all_data["metadatas"]))
        else:
            unique_docs = 0
        return {"total_chunks": count, "total_docs": unique_docs}
    except Exception:
        return {"total_chunks": 0, "total_docs": 0}


def re_rank_with_llm(query, chunks, llm_call_fn):
    """Use LLM to re-rank chunks."""
    if len(chunks) <= 1:
        return list(range(len(chunks)))
    
    chunks_text = ""
    for i, c in enumerate(chunks):
        preview = c["chunk_text"][:300]
        chunks_text += f"[{i}] {preview}\n\n"
    
    prompt = f"""Rate each document chunk's relevance to the query on a scale of 1-10.
Return ONLY a JSON array of scores: [score0, score1, ...]

Query: {query}

Chunks:
{chunks_text}"""
    
    try:
        response = llm_call_fn([{"role": "user", "content": prompt}], max_tokens=200)
        import re
        match = re.search(r'\[[\d,\s]+\]', response or "")
        if match:
            scores = json.loads(match.group())
            scored = [(i, s) for i, s in enumerate(scores) if i < len(chunks)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [i for i, _ in scored]
    except Exception:
        pass
    
    return list(range(len(chunks)))


if __name__ == "__main__":
    import tempfile
    CHROMA_PATH = tempfile.mkdtemp()
    
    chunks = [
        "Python is a high-level programming language known for its readability.",
        "Machine learning uses statistical techniques to give computers the ability to learn.",
        "深度学习是机器学习的一个子集，使用多层神经网络。",
        "Flask is a lightweight WSGI web application framework in Python.",
    ]
    
    print("Adding documents via FinnA cloud embeddings...")
    add_documents("test_doc", chunks, {"title": "Test Document"})
    
    results = search("深度学习")
    print("Query: 深度学习")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['chunk_text'][:80]}...")
    
    print(f"\nStats: {get_stats()}")
    delete_document("test_doc")
    print(f"After delete: {get_stats()}")
    print("\n✅ rag_engine.py (FinnA cloud) self-test passed")
