#!/usr/bin/env python3
"""NotebookLM-style RAG app v2 — vector search, PDF support, multi-stage podcast with TTS."""
import os, json, re, hashlib, time, gc
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

from document_processor import process_file, summarize
from rag_engine import add_documents, search, delete_document, get_stats, re_rank_with_llm
from podcast_pipeline import generate_podcast as gen_podcast

app = Flask(__name__, static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "documents")
INDEX_DIR = os.path.join(BASE_DIR, "data", "indexes")

# FinnA config
FINNA_KEY = os.environ.get("FINNA_KEY", "app-ULzJbc3OaIN50mZVSU7sAa97")
FINNA_BASE = "https://www.finna.com.cn/v1"
LLM_MODEL = "deepseek-v4-flash"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "podcasts"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "chroma_db"), exist_ok=True)

# Set ChromaDB path for rag_engine
import rag_engine
rag_engine.CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")


# ─── Document Store (lightweight JSON for doc metadata) ───
def load_docs():
    path = os.path.join(INDEX_DIR, "documents.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_docs(docs):
    with open(os.path.join(INDEX_DIR, "documents.json"), "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


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
                timeout=90
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
    return jsonify({"documents": [
        {"id": d["id"], "title": d["title"], "created": d["created"],
         "chunks": d.get("chunk_count", 0), "size": d.get("size", 0),
         "summary": d.get("summary", "")}
        for d in docs
    ]})

@app.route("/api/docs/<doc_id>", methods=["DELETE"])
def del_doc(doc_id):
    docs = load_docs()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if doc:
        filepath = os.path.join(DOCS_DIR, doc.get("filename", ""))
        if os.path.exists(filepath):
            os.remove(filepath)
        delete_document(doc_id)
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
    
    # Process document
    result = process_file(filepath, filename)
    chunks = result.get("chunks", [])
    
    if not chunks:
        return jsonify({"error": "无法解析文档内容，请检查文件格式"}), 400
    
    doc_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]
    
    # Index into vector store
    add_documents(doc_id, chunks, {"title": filename, "source": filename})
    
    # Auto-generate summary
    doc_summary = summarize(result["full_text"], 200)
    
    doc = {
        "id": doc_id,
        "title": filename,
        "filename": filename,
        "created": datetime.now().isoformat(),
        "size": len(result["full_text"]),
        "chunk_count": len(chunks),
        "summary": doc_summary,
        "full_text": result["full_text"]
    }
    
    docs = load_docs()
    docs.append(doc)
    save_docs(docs)
    
    # Generate suggested questions
    suggested = _generate_suggested_questions(result["full_text"])
    
    gc.collect()
    
    return jsonify({
        "ok": True,
        "doc": {
            "id": doc_id, "title": filename, "chunks": len(chunks),
            "summary": doc_summary, "suggested": suggested
        }
    })

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id", "")
    history = data.get("history", [])  # [{role, content}]
    
    if not query:
        return jsonify({"error": "empty query"}), 400
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    
    if not docs:
        return jsonify({"answer": "请先上传文档。", "sources": []})
    
    # Vector + BM25 hybrid search
    doc_ids = [d["id"] for d in docs]
    results = search(query, top_k=8, doc_ids=doc_ids)
    
    # LLM re-rank
    if len(results) > 3:
        ranked_indices = re_rank_with_llm(query, results, call_llm)
        results = [results[i] for i in ranked_indices if i < len(results)]
    
    if not results:
        return jsonify({"answer": "在文档中没有找到相关内容。试试换个问法？", "sources": []})
    
    # Build context with unique sources
    context_parts = []
    sources = []
    seen_titles = set()
    
    for r in results[:5]:
        title = r["metadata"].get("title", r["metadata"].get("source", "Unknown"))
        if title not in seen_titles:
            sources.append({"title": title, "id": r["metadata"].get("doc_id", "")})
            seen_titles.add(title)
        context_parts.append(f"[来源: {title}]\n{r['chunk_text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Build messages with history
    messages = []
    if history:
        for h in history[-6:]:  # Last 3 turns
            messages.append({"role": h["role"], "content": h["content"]})
    
    prompt = f"""你是一个智能研究助手。基于以下文档内容回答用户问题。

规则：
- 只能基于提供的文档内容回答
- 如果文档中没有相关信息，明确说"文档中没有提到这部分内容"
- 回答要简洁、准确，引用具体信息
- 用中文回答

文档内容：
{context}

用户问题：{query}

请回答："""
    
    messages.append({"role": "user", "content": prompt})
    answer = call_llm(messages)
    
    return jsonify({
        "answer": answer or "LLM 调用失败，请重试",
        "sources": sources
    })

@app.route("/api/podcast", methods=["POST"])
def podcast():
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    topic = data.get("topic", "").strip()
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    
    if not docs:
        return jsonify({"error": "请先上传文档"}), 400
    
    # Collect full text
    full_text = ""
    for doc in docs:
        full_text += f"\n\n## {doc['title']}\n\n"
        full_text += doc.get("full_text", "")
    
    if not topic:
        topic = docs[0]["title"] if len(docs) == 1 else "多文档综合分析"
    
    output_dir = os.path.join(BASE_DIR, "static", "podcasts")
    result = gen_podcast(full_text, topic, call_llm, output_dir)
    
    if result.get("error"):
        return jsonify({"error": result["error"]}), 500
    
    return jsonify({
        "script": result["script"],
        "segments": result["segments"],
        "audio_url": f"/static/podcasts/{os.path.basename(result['audio_path'])}" if result.get("audio_path") else None,
        "title": topic,
        "outline": result.get("outline", [])
    })

@app.route("/api/suggest", methods=["POST"])
def suggest():
    """Generate suggested questions for a document."""
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    
    if not docs:
        return jsonify({"questions": []})
    
    full_text = docs[0].get("full_text", "")
    questions = _generate_suggested_questions(full_text)
    return jsonify({"questions": questions})

@app.route("/api/health")
def health():
    stats = get_stats()
    return jsonify({"status": "ok", "docs": len(load_docs()), "chunks": stats["total_chunks"]})


def _generate_suggested_questions(text):
    """Generate 3 follow-up questions based on document content."""
    if len(text) < 100:
        return []
    
    preview = text[:3000]
    prompt = f"""基于以下文档内容，生成 3 个读者可能会问的问题。
问题要具体、有深度，能引导深入理解文档内容。
返回格式：每行一个问题，不要编号。

文档：
{preview}

3 个问题："""
    
    response = call_llm([{"role": "user", "content": prompt}], max_tokens=300)
    if not response:
        return []
    
    questions = []
    for line in response.split("\n"):
        line = line.strip()
        line = re.sub(r'^\d+[\.\)、]\s*', '', line)
        if line and len(line) > 5:
            questions.append(line)
    
    return questions[:3]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)
