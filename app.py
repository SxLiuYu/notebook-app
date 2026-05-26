#!/usr/bin/env python3
"""NotebookLM-style RAG app v2.1 — folders, cloud embeddings, PDF, podcast + TTS."""
import os, json, re, hashlib, time, gc
from datetime import datetime
from flask import Flask, request, jsonify, render_template
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

import rag_engine
rag_engine.CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")


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


# ─── Folder Store ───
def load_folders():
    path = os.path.join(INDEX_DIR, "folders.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_folders(folders):
    with open(os.path.join(INDEX_DIR, "folders.json"), "w") as f:
        json.dump(folders, f, ensure_ascii=False, indent=2)


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


# ─── Routes: Home ───
@app.route("/")
def index():
    return render_template("index.html")

# ─── Routes: Folders ───
@app.route("/api/folders", methods=["GET"])
def list_folders():
    folders = load_folders()
    docs = load_docs()
    for f in folders:
        f["doc_count"] = sum(1 for d in docs if d.get("folder_id") == f["id"])
    uncategorized = sum(1 for d in docs if not d.get("folder_id"))
    return jsonify({"folders": folders, "uncategorized": uncategorized})

@app.route("/api/folders", methods=["POST"])
def create_folder():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    folders = load_folders()
    fid = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
    folder = {"id": fid, "name": name, "created": datetime.now().isoformat()}
    folders.append(folder)
    save_folders(folders)
    return jsonify({"ok": True, "folder": folder})

@app.route("/api/folders/<fid>", methods=["PUT"])
def rename_folder(fid):
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    folders = load_folders()
    for f in folders:
        if f["id"] == fid:
            f["name"] = name
            save_folders(folders)
            return jsonify({"ok": True, "folder": f})
    return jsonify({"error": "not found"}), 404

@app.route("/api/folders/<fid>", methods=["DELETE"])
def delete_folder(fid):
    folders = load_folders()
    folders = [f for f in folders if f["id"] != fid]
    save_folders(folders)
    # Move docs to uncategorized
    docs = load_docs()
    for d in docs:
        if d.get("folder_id") == fid:
            d["folder_id"] = None
    save_docs(docs)
    return jsonify({"ok": True})

@app.route("/api/docs/<doc_id>/move", methods=["PUT"])
def move_doc(doc_id):
    data = request.get_json()
    folder_id = data.get("folder_id")  # None = uncategorized
    docs = load_docs()
    for d in docs:
        if d["id"] == doc_id:
            d["folder_id"] = folder_id
            save_docs(docs)
            return jsonify({"ok": True, "folder_id": folder_id})
    return jsonify({"error": "not found"}), 404

# ─── Routes: Documents ───
@app.route("/api/docs", methods=["GET"])
def list_docs():
    folder_id = request.args.get("folder_id", "")
    docs = load_docs()
    if folder_id == "uncategorized":
        docs = [d for d in docs if not d.get("folder_id")]
    elif folder_id:
        docs = [d for d in docs if d.get("folder_id") == folder_id]
    
    folders = load_folders()
    folder_map = {f["id"]: f["name"] for f in folders}
    
    return jsonify({"documents": [
        {"id": d["id"], "title": d["title"], "created": d["created"],
         "chunks": d.get("chunk_count", 0), "size": d.get("size", 0),
         "summary": d.get("summary", ""),
         "folder_id": d.get("folder_id"),
         "folder_name": folder_map.get(d.get("folder_id", ""), "")}
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
    
    # Get folder_id from form data
    folder_id = request.form.get("folder_id", "") or None
    
    filename = f.filename
    filepath = os.path.join(DOCS_DIR, filename)
    f.save(filepath)
    
    result = process_file(filepath, filename)
    chunks = result.get("chunks", [])
    
    if not chunks:
        return jsonify({"error": "无法解析文档内容，请检查文件格式"}), 400
    
    doc_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]
    
    add_documents(doc_id, chunks, {"title": filename, "source": filename})
    
    doc_summary = summarize(result["full_text"], 200)
    
    doc = {
        "id": doc_id,
        "title": filename,
        "filename": filename,
        "created": datetime.now().isoformat(),
        "size": len(result["full_text"]),
        "chunk_count": len(chunks),
        "summary": doc_summary,
        "full_text": result["full_text"],
        "folder_id": folder_id
    }
    
    docs = load_docs()
    docs.append(doc)
    save_docs(docs)
    
    suggested = _generate_suggested_questions(result["full_text"])
    gc.collect()
    
    return jsonify({
        "ok": True,
        "doc": {
            "id": doc_id, "title": filename, "chunks": len(chunks),
            "summary": doc_summary, "suggested": suggested,
            "folder_id": folder_id
        }
    })

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id", "")
    folder_id = data.get("folder_id", "")
    history = data.get("history", [])
    
    if not query:
        return jsonify({"error": "empty query"}), 400
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    elif folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]
    
    if not docs:
        return jsonify({"answer": "请先上传文档。", "sources": []})
    
    doc_ids = [d["id"] for d in docs]
    results = search(query, top_k=8, doc_ids=doc_ids)
    
    if len(results) > 3:
        ranked_indices = re_rank_with_llm(query, results, call_llm)
        results = [results[i] for i in ranked_indices if i < len(results)]
    
    if not results:
        return jsonify({"answer": "在文档中没有找到相关内容。试试换个问法？", "sources": []})
    
    context_parts = []
    sources = []
    seen_titles = set()
    
    for r in results[:5]:
        title = r["metadata"].get("title", r["metadata"].get("source", "Unknown"))
        if title not in seen_titles:
            sources.append({"title": title, "id": r["metadata"].get("doc_id", "")})
            seen_titles.add(title)
        context_parts.append(f"[来源: {title}]" + chr(10) + r["chunk_text"])
    
        context = "\\n\\n---\\n\\n".join(context_parts)
    
    messages = []
    if history:
        for h in history[-10:]:
            messages.append({"role": h["role"], "content": h["content"]})
    
    prompt = f"""你是一个智能研究助手。基于以下文档内容回答用户问题。

规则：
- 只能基于提供的文档内容回答
- 如果文档中没有相关信息，明确说"文档中没有提到这部分内容"
- 回答要简洁、准确，引用具体信息
- 用中文回答，使用Markdown格式（标题、列表、表格、加粗等）让回答更清晰

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
    folder_id = data.get("folder_id", "")
    topic = data.get("topic", "").strip()
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    elif folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]
    
    if not docs:
        return jsonify({"error": "请先上传文档"}), 400
    
    full_text = ""
    for doc in docs:
        full_text += chr(10) + chr(10) + "## " + doc['title'] + chr(10) + chr(10)
        full_text += doc.get("full_text", "")
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



# ─── Routes: Knowledge ───
@app.route("/api/folders/<fid>/knowledge", methods=["POST"])
def folder_knowledge(fid):
    """Generate knowledge summary, mindmap, and learning path for a folder."""
    docs = load_docs()
    if fid == "uncategorized":
        folder_docs = [d for d in docs if not d.get("folder_id")]
        folder_name = "未分类"
    else:
        folder_docs = [d for d in docs if d.get("folder_id") == fid]
        folders = load_folders()
        folder_name = next((f["name"] for f in folders if f["id"] == fid), fid)
    
    if not folder_docs:
        return jsonify({"error": "该分类下没有文档"}), 400
    
    # Collect all document content with titles
    full_text = ""
    for d in folder_docs:
        full_text += f"\n\n## {d['title']}\n\n"
        full_text += d.get("full_text", "")[:5000]  # Cap per doc
    
    full_text = full_text[:15000]  # Total cap
    
    # Step 1: Generate structured markdown summary
    summary_prompt = f"""你是一个知识整理专家。基于以下文档内容，生成一份结构化的知识梳理（Markdown格式）。

要求：
- 使用多级标题（# ## ###）
- 提取核心概念、关键知识点
- 用表格对比重要概念
- 列出文档中提到的所有关键术语
- 控制在500字以内，精炼

文档内容：
{full_text}

知识梳理："""
    
    summary_md = call_llm([{"role": "user", "content": summary_prompt}], max_tokens=1200)
    
    # Step 2: Generate mindmap (JSON tree)
    mindmap_prompt = f"""基于以下文档内容，生成一个知识树（思维导图）结构，JSON格式。

格式要求：
{{"topic": "主题名称", "children": [{{"topic": "子主题", "children": [...]}}, ...]}}
最多3层深度，每个节点topic简短（不超过10字）。只返回JSON，不要解释。

文档内容：
{full_text[:5000]}

JSON:"""

    mindmap_json = None
    mindmap_raw = call_llm([{"role": "user", "content": mindmap_prompt}], max_tokens=800)
    if mindmap_raw:
        import re
        match = re.search(r'\{[\s\S]*\}', mindmap_raw)
        if match:
            try:
                mindmap_json = json.loads(match.group())
            except:
                mindmap_json = None
    
    # Step 3: Generate learning path
    path_prompt = f"""你是一个学习路径设计师。基于以下文档内容，设计一条结构化的学习路径。

要求：
- 分3-5个阶段
- 每阶段包含：阶段名称、学习目标、核心知识点（列表）、建议用时
- 用Markdown格式返回
- 如果文档涉及实操技能，加入练习建议

文档内容：
{full_text[:6000]}

学习路径："""
    
    learning_path = call_llm([{"role": "user", "content": path_prompt}], max_tokens=1000)
    
    return jsonify({
        "folder_name": folder_name,
        "summary_md": summary_md or "生成失败",
        "mindmap": mindmap_json,
        "learning_path": learning_path or "生成失败",
        "doc_count": len(folder_docs)
    })

@app.route("/api/health")
def health():
    stats = get_stats()
    return jsonify({"status": "ok", "docs": len(load_docs()), "chunks": stats["total_chunks"]})


def _generate_suggested_questions(text):
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
    for line in response.split(chr(10)):
        line = line.strip()
        line = re.sub(r'^\d+[\.\)、]\s*', '', line)
        if line and len(line) > 5:
            questions.append(line)
    
    return questions[:3]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)