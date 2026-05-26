#!/usr/bin/env python3
"""NotebookLM-style RAG app — 文档上传、问答、播客生成"""
import os, json, re, hashlib, time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

app = Flask(__name__, static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "documents")
INDEX_DIR = os.path.join(BASE_DIR, "data", "indexes")

# FinnA config
FINNA_KEY = "app-ULzJbc3OaIN50mZVSU7sAa97"
FINNA_BASE = "https://www.finna.com.cn/v1"
LLM_MODEL = "deepseek-v4-flash"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ─── Document Store ───
def load_docs():
    path = os.path.join(INDEX_DIR, "documents.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_docs(docs):
    with open(os.path.join(INDEX_DIR, "documents.json"), "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for retrieval."""
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < chunk_size:
            current += p + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    
    # If chunks are too small, merge; if too large, split
    result = []
    for c in chunks:
        if len(c) > chunk_size * 2:
            # Split by sentences
            sentences = re.split(r'(?<=[。！？.!?])\s*', c)
            sub = ""
            for s in sentences:
                if len(sub) + len(s) < chunk_size:
                    sub += s
                else:
                    if sub:
                        result.append(sub.strip())
                    sub = s
            if sub:
                result.append(sub.strip())
        else:
            result.append(c)
    return [r for r in result if len(r) > 20]

def search_chunks(docs, query, top_k=5):
    """Search across document chunks with Chinese-aware matching."""
    query_lower = query.lower()
    # For CJK text, use substring sliding window
    # For English/spaced text, use word-level matching
    scored = []
    for doc in docs:
        for i, chunk in enumerate(doc.get("chunks", [])):
            chunk_lower = chunk.lower()
            score = 0
            # Direct substring match
            if query_lower in chunk_lower:
                score += 10
            # Sliding window: check if significant parts of query match
            # For Chinese: use 2-gram and 3-gram matching
            if any(ord(c) > 0x2000 for c in query):  # has CJK characters
                for n in [3, 2]:
                    for j in range(len(query) - n + 1):
                        ngram = query[j:j+n]
                        if ngram in chunk_lower:
                            score += 1
            # Word-level for English
            query_words = query_lower.split()
            score += sum(2 for w in query_words if len(w) > 1 and w in chunk_lower)
            # Bonus for title match
            if query_lower in doc["title"].lower() or any(
                len(w) > 1 and w in doc["title"].lower() for w in query_lower.split()
            ):
                score += 3
            if score > 0:
                scored.append((score, doc, i, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# ─── LLM Call ───
def call_llm(messages, max_tokens=2048):
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{FINNA_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {FINNA_KEY}", "Content-Type": "application/json"},
                json={"model": LLM_MODEL, "messages": messages, "temperature": 0.3,
                      "max_tokens": max_tokens, "stream": False,
                      "extra_body": {"enable_thinking": False}},
                timeout=60
            )
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

# ─── Routes ───
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/docs", methods=["GET"])
def list_docs():
    docs = load_docs()
    return jsonify({"documents": [{"id": d["id"], "title": d["title"], "created": d["created"],
                                    "chunks": len(d.get("chunks", [])), "size": d.get("size", 0)} for d in docs]})

@app.route("/api/docs/<doc_id>", methods=["DELETE"])
def delete_doc(doc_id):
    docs = load_docs()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if doc:
        # Delete file
        filepath = os.path.join(DOCS_DIR, doc.get("filename", ""))
        if os.path.exists(filepath):
            os.remove(filepath)
        docs = [d for d in docs if d["id"] != doc_id]
        save_docs(docs)
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "no filename"}), 400
    
    filename = f.filename
    filepath = os.path.join(DOCS_DIR, filename)
    f.save(filepath)
    
    # Read and chunk text
    try:
        text = open(filepath, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        text = open(filepath, "r", encoding="gbk", errors="ignore").read()
    
    chunks = chunk_text(text)
    
    doc_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]
    doc = {
        "id": doc_id,
        "title": filename,
        "filename": filename,
        "created": datetime.now().isoformat(),
        "size": len(text),
        "chunks": chunks
    }
    
    docs = load_docs()
    docs.append(doc)
    save_docs(docs)
    
    return jsonify({"ok": True, "doc": {"id": doc_id, "title": filename, "chunks": len(chunks)}})

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id", "")  # optional: limit to one doc
    
    if not query:
        return jsonify({"error": "empty query"}), 400
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    
    if not docs:
        return jsonify({"answer": "请先上传文档。", "sources": []})
    
    # Search
    results = search_chunks(docs, query, top_k=5)
    
    if not results:
        return jsonify({"answer": "在文档中没有找到相关内容。试试换个问法？", "sources": []})
    
    # Build context
    context_parts = []
    sources = []
    seen = set()
    for score, doc, chunk_idx, chunk_text in results:
        source_key = f"{doc['title']}"
        if source_key not in seen:
            sources.append({"title": doc["title"], "id": doc["id"]})
            seen.add(source_key)
        context_parts.append(f"[来源: {doc['title']}]\n{chunk_text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""你是一个智能研究助手。基于以下文档内容回答用户问题。

如果文档中没有相关信息，请明确说"文档中没有提到这部分内容"。
回答要简洁、准确，引用具体信息。

文档内容：
{context}

用户问题：{query}

请回答："""
    
    answer = call_llm([{"role": "user", "content": prompt}])
    
    return jsonify({
        "answer": answer or "LLM 调用失败，请重试",
        "sources": sources
    })

@app.route("/api/podcast", methods=["POST"])
def generate_podcast():
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    topic = data.get("topic", "文档内容概览")
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    
    if not docs:
        return jsonify({"error": "请先上传文档"}), 400
    
    # Collect full text
    full_text = ""
    for doc in docs:
        full_text += f"\n\n## {doc['title']}\n\n"
        full_text += "\n\n".join(doc.get("chunks", []))
    
    if len(full_text) > 8000:
        full_text = full_text[:8000] + "\n...(内容已截断)"
    
    prompt = f"""你是一个播客主持人。请基于以下文档内容，生成一段 5-8 分钟的播客对话脚本。

格式要求：
- 两个角色：主持人A（好奇提问者）和专家B（深度解读）
- 用中文对话
- 自然的口语表达，不要太书面
- 标记角色：[A] 和 [B]
- 控制在 1500-2500 字

主题：{topic}

文档内容：
{full_text}

请生成播客对话脚本："""
    
    script = call_llm([{"role": "user", "content": prompt}], max_tokens=4096)
    
    if not script:
        return jsonify({"error": "生成失败"}), 500
    
    # Parse script into segments
    segments = []
    for line in script.split("\n"):
        line = line.strip()
        if line.startswith("[A]") or line.startswith("[B]"):
            role = "host" if line.startswith("[A]") else "expert"
            text = line[3:].strip()
            if text:
                segments.append({"role": role, "text": text})
    
    return jsonify({
        "script": script,
        "segments": segments,
        "title": topic
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "docs": len(load_docs())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)
