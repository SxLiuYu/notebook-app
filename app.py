#!/usr/bin/env python3
"""NotebookLM-style RAG app v2.1 — folders, cloud embeddings, PDF, podcast + TTS."""
import os, json, re, hashlib, time, gc
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import requests

from document_processor import process_file, summarize, is_archive, extract_archive, is_media, is_video, get_ext
import shutil as _shutil
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


@app.route("/whitelist")
def whitelist_page():
    """403 redirect target — show IP whitelist request page."""
    ip = request.args.get("ip", request.remote_addr or "unknown")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>访问受限 — Notebook</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; padding: 20px;
}}
.card {{
    background: #1e293b; border-radius: 16px; padding: 40px;
    max-width: 440px; width: 100%; text-align: center;
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}}
.icon {{ font-size: 56px; margin-bottom: 16px; }}
h1 {{ font-size: 22px; margin-bottom: 8px; }}
.ip-box {{
    background: #0f172a; border-radius: 8px; padding: 12px 20px;
    margin: 20px 0; font-family: monospace; font-size: 16px;
    color: #38bdf8; letter-spacing: 1px;
}}
p {{ color: #94a3b8; font-size: 14px; line-height: 1.6; margin-bottom: 24px; }}
.btn {{
    display: inline-block; background: #3b82f6; color: white;
    border: none; border-radius: 10px; padding: 14px 32px;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: background 0.2s; text-decoration: none;
}}
.btn:hover {{ background: #2563eb; }}
.btn:disabled {{ background: #475569; cursor: not-allowed; }}
#msg {{ margin-top: 16px; font-size: 14px; min-height: 24px; }}
.success {{ color: #34d399; }}
.error {{ color: #f87171; }}
</style>
</head>
<body>
<div class="card">
    <div class="icon">🔒</div>
    <h1>IP 未在白名单中</h1>
    <div class="ip-box">{ip}</div>
    <p>你的设备 IP 不在访问白名单中。<br>点击下方按钮申请加白，管理员审核后即可访问。</p>
    <button class="btn" onclick="requestAccess()" id="btn">📩 申请加白</button>
    <div id="msg"></div>
</div>
<script>
async function requestAccess() {{
    const btn = document.getElementById('btn');
    const msg = document.getElementById('msg');
    btn.disabled = true;
    btn.textContent = '⏳ 提交中…';
    msg.className = '';
    msg.textContent = '';
    try {{
        const r = await fetch('/api/whitelist/add', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{ip: '{ip}'}})
        }});
        const data = await r.json();
        if (r.ok) {{
            msg.className = 'success';
            msg.textContent = data.message || '✅ 加白成功！请刷新页面';
            btn.textContent = '🔄 刷新页面';
            btn.onclick = () => location.href = '/';
        }} else {{
            msg.className = 'error';
            msg.textContent = data.error || '提交失败';
            btn.disabled = false;
            btn.textContent = '📩 申请加白';
        }}
    }} catch(e) {{
        msg.className = 'error';
        msg.textContent = '网络错误: ' + e.message;
        btn.disabled = false;
        btn.textContent = '📩 申请加白';
    }}
}}
</script>
</body>
</html>"""


@app.route("/api/whitelist/add", methods=["POST"])
def whitelist_add():
    """Add an IP to nginx whitelist and reload."""
    from flask import request, jsonify
    import subprocess, re

    data = request.get_json(silent=True) or {}
    ip = data.get("ip", "").strip()
    if not ip or not re.match(r'^[\d.]+$', ip):
        return jsonify({"error": "无效的 IP 地址"}), 400

    nginx_conf = "/etc/nginx/sites-available/notebook"
    try:
        with open(nginx_conf, 'r') as f:
            config = f.read()

        # Check if already whitelisted
        if f"allow {ip};" in config:
            return jsonify({"message": f"IP {ip} 已在白名单中，请刷新页面"})

        # Insert allow directive before "deny all;"
        new_config = config.replace("deny all;", f"allow {ip};\n        deny all;")

        with open(nginx_conf, 'w') as f:
            f.write(new_config)

        # Test and reload nginx
        result = subprocess.run(["nginx", "-t"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # Rollback
            with open(nginx_conf, 'w') as f:
                f.write(config)
            return jsonify({"error": f"nginx 配置验证失败: {result.stderr}"}), 500

        subprocess.run(["systemctl", "reload", "nginx"], timeout=10)
        return jsonify({"message": f"✅ IP {ip} 已加白！请刷新页面访问"})

    except Exception as e:
        return jsonify({"error": f"加白失败: {str(e)}"}), 500
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
         "folder_name": folder_map.get(d.get("folder_id", ""), ""),
         "is_media": d.get("is_media", False),
         "is_video": d.get("is_video", False)}
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
    folder_id = request.form.get("folder_id", "") or None
    filename = f.filename
    filepath = os.path.join(DOCS_DIR, filename)
    f.save(filepath)
    
    # ARCHIVE handling
    if is_archive(filename):
        extracted_files, tmpdir = extract_archive(filepath, filename)
        uploaded = []
        errors = []
        for ef in extracted_files:
            try:
                result = process_file(ef["filepath"], ef["filename"])
                chunks = result.get("chunks", [])
                if not chunks:
                    errors.append({"file": ef["filename"], "error": "unparseable"})
                    continue
                doc_id = hashlib.md5(f"{ef['filename']}{time.time()}".encode()).hexdigest()[:12]
                add_documents(doc_id, chunks, {"title": ef["filename"], "source": ef["filename"], "doc_id": doc_id})
                doc_summary = summarize(result["full_text"], 200)
                doc = {
                    "id": doc_id, "title": ef["filename"], "filename": ef["filename"],
                    "created": datetime.now().isoformat(), "size": len(result["full_text"]),
                    "chunk_count": len(chunks), "summary": doc_summary,
                    "full_text": result["full_text"], "folder_id": folder_id,
                    "is_media": is_media(ef["filename"]),
                    "is_video": is_video(ef["filename"])
                }
                docs = load_docs(); docs.append(doc); save_docs(docs)
                uploaded.append({"id": doc_id, "title": ef["filename"], "chunks": len(chunks), "summary": doc_summary, "folder_id": folder_id})
            except Exception as e:
                errors.append({"file": ef["filename"], "error": str(e)})
        _shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()
        return jsonify({"ok": True, "archive": True, "archive_name": filename, "files": uploaded, "errors": errors, "total": len(uploaded)})
    
    # Single file
    result = process_file(filepath, filename)
    chunks = result.get("chunks", [])
    if not chunks:
        return jsonify({"error": "unparseable"}), 400
    doc_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:12]
    add_documents(doc_id, chunks, {"title": filename, "source": filename, "doc_id": doc_id})
    doc_summary = summarize(result["full_text"], 200)
    is_m = is_media(filename)
    media_url = None
    if is_m:
        media_dir = os.path.join(BASE_DIR, "static", "media")
        os.makedirs(media_dir, exist_ok=True)
        dest = os.path.join(media_dir, filename)
        if not os.path.exists(dest):
            _shutil.copy2(filepath, dest)
        media_url = f"/static/media/{filename}"
    doc = {
        "id": doc_id, "title": filename, "filename": filename,
        "created": datetime.now().isoformat(), "size": len(result["full_text"]),
        "chunk_count": len(chunks), "summary": doc_summary,
        "full_text": result["full_text"], "folder_id": folder_id,
        "is_media": is_m, "is_video": is_video(filename) if is_m else False,
        "media_url": media_url
    }
    docs = load_docs(); docs.append(doc); save_docs(docs)
    suggested = _generate_suggested_questions(result["full_text"])
    gc.collect()
    return jsonify({"ok": True, "doc": {"id": doc_id, "title": filename, "chunks": len(chunks), "summary": doc_summary, "suggested": suggested, "is_media": is_m, "is_video": is_video(filename) if is_m else False, "media_url": media_url, "folder_id": folder_id}})

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
    
    if not doc_id and not folder_id:
        return jsonify({"error": "请先选择文档或分类"}), 400
    
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
        for h in history[-6:]:
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
    
    if not doc_id and not folder_id:
        return jsonify({"error": "请先选择文档或分类"}), 400
    
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
    
    if not doc_id and not folder_id:
        return jsonify({"error": "请先选择文档或分类"}), 400
    
    if not docs:
        return jsonify({"questions": []})
    
    full_text = docs[0].get("full_text", "")
    questions = _generate_suggested_questions(full_text)
    return jsonify({"questions": questions})


@app.route("/api/knowledge", methods=["POST"])
def knowledge():
    """Extract structured knowledge from selected documents."""
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    folder_id = data.get("folder_id", "")
    
    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    elif folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]
    
    if not doc_id and not folder_id:
        return jsonify({"error": "请先选择文档或分类"}), 400
    
    if not docs:
        return jsonify({"error": "请先选择文档"}), 400
    
    doc_ids = [d["id"] for d in docs]
    all_chunks = search("主要概念 核心知识点 关键术语 方法论 框架 定义 分类 总结", top_k=20, doc_ids=doc_ids)
    
    if not all_chunks:
        return jsonify({"error": "文档中没有找到可梳理的内容"}), 404
    
    doc_map = {}
    for r in all_chunks[:15]:
        title = r["metadata"].get("title", "Unknown")
        if title not in doc_map:
            doc_map[title] = []
        doc_map[title].append(r["chunk_text"])
    
    context_parts = []
    for title, chunks in doc_map.items():
        context_parts.append(f"## {title}\n" + "\n\n".join(chunks))
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""基于以下文档，提取知识结构。用中文，简洁扼要。

## 核心概念
- **术语**：一句话解释

## 主题结构
### 主题
- 要点

## 关键结论
1. 结论

文档：
{context}"""

    messages = [{"role": "user", "content": prompt}]
    answer = call_llm(messages, max_tokens=1536)
    
    sections = {"concepts": [], "topics": [], "conclusions": [], "raw": answer or ""}
    
    if answer:
        concept_section = False
        topic_section = False
        conclusion_section = False
        current_topic = None
        
        for line in answer.split("\n"):
            stripped = line.strip()
            if "核心概念" in stripped or "关键概念" in stripped:
                concept_section = True; topic_section = False; conclusion_section = False
                continue
            elif "主题结构" in stripped or "知识结构" in stripped:
                concept_section = False; topic_section = True; conclusion_section = False
                continue
            elif "关键结论" in stripped or "重要结论" in stripped or "总结" in stripped:
                concept_section = False; topic_section = False; conclusion_section = True
                continue
            
            if concept_section and stripped.startswith("- **"):
                sections["concepts"].append(stripped)
            elif topic_section:
                if stripped.startswith("### "):
                    current_topic = stripped.replace("### ", "").strip()
                    sections["topics"].append({"title": current_topic, "items": []})
                elif stripped.startswith("- ") and current_topic:
                    if sections["topics"]:
                        sections["topics"][-1]["items"].append(stripped[2:])
            elif conclusion_section and stripped and stripped[0].isdigit():
                sections["conclusions"].append(stripped)
    
    return jsonify(sections)


@app.route("/api/exam", methods=["POST"])
def exam():
    """生成考点梳理 + 自测题（选择题/填空题/简答题）"""
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    folder_id = data.get("folder_id", "")
    question_type = data.get("type", "all")  # all, choice, fill, essay
    count = data.get("count", 10)

    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    elif folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]

    if not docs:
        return jsonify({"error": "请先选择文档或分类"}), 400

    # Gather content for exam generation
    doc_ids = [d["id"] for d in docs]
    # Search for key knowledge points first
    key_chunks = search("核心概念 定义 公式 考点 重点 原理 方法 步骤 分类 区别 特点 作用 意义",
                        top_k=15, doc_ids=doc_ids)

    if not key_chunks:
        key_chunks = search("内容 说明 介绍", top_k=10, doc_ids=doc_ids)

    context_parts = []
    for r in key_chunks[:12]:
        title = r["metadata"].get("title", r["metadata"].get("source", ""))
        context_parts.append(f"[{title}]\n{r['chunk_text']}")

    context = "\n\n---\n\n".join(context_parts)

    # Generate exam based on type
    type_desc = {
        "all": "选择题(4题)、填空题(3题)、简答题(3题)",
        "choice": f"选择题({count}题)",
        "fill": f"填空题({count}题)",
        "essay": f"简答题({count}题)"
    }.get(question_type, f"选择题(4题)、填空题(3题)、简答题(3题)")

    prompt = f"""你是资深考试出题专家。基于以下文档内容，生成一份考点梳理和自测题。

## 要求

### 第一部分：考点梳理
列出文档中的核心考点，按重要性排序，每个考点标注：
- ★★★ 高频必考
- ★★ 常考
- ★ 了解即可

### 第二部分：自测题
生成{type_desc}。格式如下：

**选择题**（每题4个选项，标注正确答案）：
```
1. 题目？
A. 选项  B. 选项  C. 选项  D. 选项
答案：X
解析：一句话解释
```

**填空题**：
```
1. ______ 是XXX的核心概念。
答案：XXX
```

**简答题**：
```
1. 题目？
参考答案：XXX
```

注意：
- 题目必须基于文档内容，不能凭空编造
- 选择题选项要有迷惑性
- 用中文出题
- 先出考点梳理，再出自测题

文档内容：
{context}

请生成："""

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(messages, max_tokens=3072)

    if not result:
        return jsonify({"error": "生成失败，请重试"}), 500

    # Parse result
    return jsonify({
        "raw": result,
        "title": docs[0]["title"] if docs else "文档"
    })



@app.route("/api/mindmap", methods=["POST"])
def mindmap():
    """生成思维导图（Mermaid.js 格式）"""
    data = request.get_json()
    doc_id = data.get("doc_id", "")
    folder_id = data.get("folder_id", "")
    depth = data.get("depth", 3)  # 层级深度

    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d["id"] == doc_id]
    elif folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]

    if not docs:
        return jsonify({"error": "请先选择文档或分类"}), 400

    doc_ids = [d["id"] for d in docs]
    key_chunks = search("主题 分类 概述 结构 框架 概念 方法 过程 步骤 原理",
                        top_k=12, doc_ids=doc_ids)

    context_parts = []
    for r in key_chunks[:10]:
        title = r["metadata"].get("title", "")
        context_parts.append(f"[{title}]\n{r['chunk_text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""基于以下文档内容，生成 Mermaid.js mindmap 格式的思维导图。

要求：
1. 层级不超过{depth}层
2. 中心主题用文档标题
3. 只输出 Mermaid 代码块，不要其他文字
4. 使用中文
5. 每个节点简洁，不超过15个字
6. 连线使用标准的 mindmap 语法

示例格式：
```mermaid
mindmap
  root((中心主题))
    一级分支
      二级知识
        三级细节
      二级知识
    一级分支
      二级知识
```

文档内容：
{context}

请生成Mermaid思维导图："""

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(messages, max_tokens=2048)

    if not result:
        return jsonify({"error": "生成失败，请重试"}), 500

    # Extract mermaid code
    mermaid_code = result
    if "```mermaid" in result:
        start = result.find("```mermaid") + len("```mermaid")
        end = result.find("```", start)
        if end > start:
            mermaid_code = result[start:end].strip()
    elif "```" in result:
        start = result.find("```") + 3
        end = result.find("```", start)
        if end > start:
            mermaid_code = result[start:end].strip()

    return jsonify({
        "mermaid": mermaid_code,
        "title": docs[0]["title"] if docs else "文档",
        "raw": result
    })



@app.route("/api/feynman", methods=["POST"])
def feynman():
    """费曼复述：挑关键段落 → 用户解释 → AI判断理解是否到位"""
    data = request.get_json()
    action = data.get("action", "pick")  # pick | evaluate
    doc_id = data.get("doc_id", "")
    folder_id = data.get("folder_id", "")

    if action == "pick":
        docs = load_docs()
        if doc_id:
            docs = [d for d in docs if d["id"] == doc_id]
        elif folder_id:
            if folder_id == "uncategorized":
                docs = [d for d in docs if not d.get("folder_id")]
            else:
                docs = [d for d in docs if d.get("folder_id") == folder_id]

        if not docs:
            return jsonify({"error": "请先选择文档或分类"}), 400

        doc_ids = [d["id"] for d in docs]

        blind = data.get("blind", False)  # 盲复述模式：不显示原文

        # Load user's progress for this doc
        progress_path = os.path.join(INDEX_DIR, "progress.json")
        progress = {}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                progress = json.load(f)

        weak_concepts = []
        if doc_id and doc_id in progress:
            weak_concepts = progress[doc_id].get("weak_concepts", [])

        # Smart pick: prioritize weak concepts
        candidates = None
        if weak_concepts:
            # Search for passages related to weak concepts
            weak_query = " ".join(weak_concepts[:3])
            candidates = search(weak_query, top_k=5, doc_ids=doc_ids)
            if not candidates or len(candidates[0]["chunk_text"]) < 60:
                candidates = None  # Fall through to default

        if not candidates:
            # Default: pick substantive passages
            candidates = search("原理 过程 方法 关系 区别 原因 论证 分析 所以 因此 例如 具体来说",
                                top_k=8, doc_ids=doc_ids)
            if not candidates:
                candidates = search("内容 说明 介绍", top_k=5, doc_ids=doc_ids)

        if not candidates:
            return jsonify({"error": "文档中没有足够长的段落用于复述"}), 404

        best = max(candidates, key=lambda x: len(x["chunk_text"]))
        passage = best["chunk_text"].strip()
        source_title = best["metadata"].get("title", best["metadata"].get("source", ""))

        return jsonify({
            "passage": passage,
            "source": source_title,
            "length": len(passage),
            "blind": blind,
            "hint": f"这段内容涉及: {', '.join(weak_concepts[:2])}" if weak_concepts and not blind else "",
            "targeted": bool(weak_concepts)
        })

    elif action == "evaluate":
        passage = data.get("passage", "").strip()
        explanation = data.get("explanation", "").strip()
        blind = data.get("blind", False)
        doc_id_eval = data.get("doc_id", "")

        if not passage or not explanation:
            return jsonify({"error": "缺少原文或你的解释"}), 400

        if len(explanation) < 10:
            return jsonify({"error": "解释太短，请至少写一句话"}), 400

        prompt = f"""你是费曼学习法的教练。你的任务是评估学习者对一段原文的理解程度。

## 原文
{passage}

## 学习者的解释
{explanation}

## 评估要求

请从以下几个维度评估，用中文回复：

### 理解判断
判定：✅ 理解正确 / ⚠️ 部分正确 / ❌ 理解有误

### 做得好的地方
- 如果理解正确，说明学习者抓住了什么核心
- 如果部分正确，指出哪部分是对的

### 遗漏了什么
- 原文中哪些重要信息没有被提到？
- 是否有隐含的前提条件被忽略了？

### 偏差在哪里
- 学习者的解释和原文有哪些不一致？
- 是否有过度推断或主观臆断？

### 薄弱概念
- 如果有理解错误，指出哪些概念需要重新学习（用「」标注，如「过拟合」）
- 如果理解正确，这部分留空

### 一句话建议
- 如果要真正理解这段内容，应该重点关注什么？

注意：
- 语气要像一位耐心的老师，鼓励为主，精准指出问题
- 不要长篇大论，每个维度 2-3 句即可
- 用中文"""

        messages = [{"role": "user", "content": prompt}]
        result = call_llm(messages, max_tokens=1536)

        if not result:
            return jsonify({"error": "评估失败，请重试"}), 500

        verdict = "⚠️ 部分正确"
        if "理解正确" in result:
            verdict = "✅ 理解正确"
        elif "理解有误" in result:
            verdict = "❌ 理解有误"

        # Extract weak concepts for progress tracking
        weak = []
        for match in re.finditer(r'「([^」]+)」', result):
            concept = match.group(1)
            if len(concept) > 2 and len(concept) < 30:
                weak.append(concept)

        return jsonify({
            "verdict": verdict,
            "feedback": result,
            "passage": passage,
            "explanation": explanation,
            "blind": blind,
            "weak_concepts": weak[:5],
            "is_correct": "正确" in verdict
        })

        # Auto-generate review items from weak concepts
        if weak and doc_id_eval:
            try:
                doc_title = ""
                docs = load_docs()
                for d in docs:
                    if d["id"] == doc_id_eval:
                        doc_title = d["title"]
                        break
                auto_generate_review_items(doc_id_eval, weak[:5], doc_title, passage[:500])
            except:
                pass

    return jsonify({"error": "无效的 action"}), 400



@app.route("/api/link", methods=["POST"])
def link():
    """跨文档知识关联：发现不同文档中概念之间的联系"""
    data = request.get_json()
    folder_id = data.get("folder_id", "")

    docs = load_docs()
    if folder_id:
        if folder_id == "uncategorized":
            docs = [d for d in docs if not d.get("folder_id")]
        else:
            docs = [d for d in docs if d.get("folder_id") == folder_id]

    if len(docs) < 2:
        return jsonify({"error": "至少需要 2 份文档才能做知识关联。请上传更多文档或选择一个分类"}), 400

    # Phase 1: Extract key concepts from each document
    doc_concepts = []
    for doc in docs:
        # Search for substantive content in each doc
        chunks = search("核心 概念 定义 原理 方法 框架 模型 理论 公式 分类 特点 作用",
                        top_k=5, doc_ids=[doc["id"]])
        if not chunks:
            chunks = search("内容 说明 介绍", top_k=3, doc_ids=[doc["id"]])
        if chunks:
            text = "\n".join([c["chunk_text"][:500] for c in chunks[:3]])
            doc_concepts.append({
                "id": doc["id"],
                "title": doc["title"],
                "excerpt": text[:1500]
            })

    if len(doc_concepts) < 2:
        return jsonify({"error": "文档中没有足够的内容用于关联分析"}), 404

    # Phase 2: Build context for LLM
    context_parts = []
    for dc in doc_concepts:
        context_parts.append(f"## [{dc['title']}]\n{dc['excerpt']}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是知识图谱专家。分析以下多份文档，发现它们之间的深层关联。

**关键原则：只找真实存在的关联，不要强行凑。如果文档之间共同点很少，诚实地说"未发现实质性关联"，千万不要编造。**

## 文档内容
{context}

## 要求

找出文档之间的**实质性关联**（不是表面的"都讲了XX"），包括：

1. **同一概念的不同表述**：A 文档的 X = B 文档的 Y（不同说法，同一件事）
2. **互补关系**：A 文档讲了理论，B 文档给了实践案例
3. **层级关系**：A 文档是 B 文档的基础/前提
4. **矛盾或张力**：A 文档和 B 文档对同一问题的观点不同
5. **意外关联**：表面无关但内在逻辑相通的概念

## 输出格式

用中文，每条关联一行，格式：
```
[文档A] 的「概念A」← 关系类型 → [文档B] 的「概念B」
简短说明：一句话解释为什么有关联
```

示例：
```
[机器学习入门] 的「过拟合」← 同一概念 → [深度学习实战] 的「泛化能力不足」
简短说明：两个文档用不同术语描述了同一个现象
```

注意：
- 只找有实质意义的关联，不要凑数
- 找不到真实关联就说找不到，不要编造
- 至少找 3 条，最多 8 条
- 每条关联必须具体到文档和概念"""

    messages = [{"role": "user", "content": prompt}]
    result = call_llm(messages, max_tokens=2048)

    if not result:
        return jsonify({"error": "分析失败，请重试"}), 500

    # Parse connections
    connections = []
    for line in result.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("注意"):
            continue
        if "←" in line and "→" in line:
            # Parse the connection line
            conn = {"raw": line}
            # Try to extract doc names and concepts
            match = re.match(r'\[([^\]]+)\]\s*的\s*「([^」]+)」\s*←\s*(.+?)\s*→\s*\[([^\]]+)\]\s*的\s*「([^」]+)」', line)
            if match:
                conn = {
                    "source_doc": match.group(1),
                    "source_concept": match.group(2),
                    "relation": match.group(3).strip(),
                    "target_doc": match.group(4),
                    "target_concept": match.group(5),
                    "raw": line
                }
            connections.append(conn)
            continue
        if line.startswith("简短说明：") and connections:
            connections[-1]["explanation"] = line.replace("简短说明：", "").strip()

    # If parsing failed, use raw
    if not connections:
        connections = [{"raw": result, "fallback": True}]

    return jsonify({
        "connections": connections,
        "doc_count": len(docs),
        "raw": result
    })



@app.route("/api/progress", methods=["GET", "POST"])
def progress():
    """学习进度存取：记住每份文档的掌握状态、费曼历史、错题"""
    path = os.path.join(INDEX_DIR, "progress.json")

    def load_progress():
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def save_progress(p):
        with open(path, "w") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)

    if request.method == "GET":
        docs = load_docs()
        progress = load_progress()

        # Enrich doc list with progress data
        enriched = []
        for d in docs:
            p = progress.get(d["id"], {})
            enriched.append({
                "id": d["id"],
                "title": d["title"],
                "created": d["created"],
                "chunks": d.get("chunk_count", 0),
                "mastered": p.get("mastered", 0),       # 0-100
                "feynman_count": p.get("feynman_count", 0),
                "feynman_correct": p.get("feynman_correct", 0),
                "exam_score": p.get("exam_score"),
                "weak_concepts": p.get("weak_concepts", []),
                "last_accessed": p.get("last_accessed", "")
            })

        return jsonify({"progress": enriched})

    elif request.method == "POST":
        data = request.get_json()
        doc_id = data.get("doc_id", "")
        update = data.get("update", {})

        if not doc_id:
            return jsonify({"error": "doc_id required"}), 400

        progress = load_progress()
        entry = progress.get(doc_id, {
            "mastered": 0,
            "feynman_count": 0,
            "feynman_correct": 0,
            "exam_score": None,
            "weak_concepts": [],
            "mastered_concepts": [],
            "feynman_history": [],
            "exam_errors": [],
            "last_accessed": ""
        })

        # Apply updates
        for key, val in update.items():
            if key in ("feynman_count", "feynman_correct"):
                entry[key] = entry.get(key, 0) + val
            elif key == "exam_score":
                entry[key] = val
            elif key in ("weak_concepts", "mastered_concepts"):
                # Merge lists, deduplicate
                existing = entry.get(key, [])
                new_items = [item for item in val if item not in existing]
                for item in new_items:
                    existing.append(item)
                entry[key] = existing[-20:]  # Keep last 20
                # Auto-generate review cards for new weak concepts
                if key == "weak_concepts" and new_items:
                    try:
                        title = ""
                        for d in load_docs():
                            if d["id"] == doc_id:
                                title = d["title"]; break
                        auto_generate_review_items(doc_id, new_items, title)
                    except:
                        pass
            elif key == "feynman_history":
                existing = entry.get(key, [])
                existing.append(val)
                entry[key] = existing[-30:]  # Keep last 30
            elif key == "exam_errors":
                existing = entry.get(key, [])
                existing.append(val)
                entry[key] = existing[-50:]
                # Auto-generate review card from exam error
                try:
                    title = ""
                    for d in load_docs():
                        if d["id"] == doc_id:
                            title = d["title"]; break
                    auto_generate_exam_review(doc_id, [val], title)
                except:
                    pass
            elif key == "mastered":
                entry[key] = val

        entry["last_accessed"] = datetime.now().isoformat()
        progress[doc_id] = entry
        save_progress(progress)

        return jsonify({"ok": True, "progress": entry})


@app.route("/api/media/<doc_id>")
def media_view(doc_id):
    """Get media metadata + transcript + AI analysis for side-by-side view."""
    docs = load_docs()
    doc = None
    for d in docs:
        if d["id"] == doc_id:
            doc = d
            break
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    if not doc.get("is_media"):
        return jsonify({"error": "Not a media file"}), 400
    
    full_text = doc.get("full_text", "")
    is_vid = doc.get("is_video", False)
    
    # Generate AI analysis of the transcript
    analysis = ""
    if full_text and len(full_text) > 100:
        try:
            type_label = "video" if is_vid else "audio"
            prompt = f"You are a content analyst. Below is a transcript from a {type_label} file. Provide a concise analysis in Chinese: 1) core theme, 2) 3-5 key points, 3) timeline markers if any.\n\nTranscript:\n{full_text[:4000]}"
            messages = [{"role": "user", "content": prompt}]
            analysis = call_llm(messages, max_tokens=1024) or ""
        except:
            analysis = ""
    
    return jsonify({
        "id": doc["id"],
        "title": doc["title"],
        "is_video": is_vid,
        "media_url": doc.get("media_url", "/static/media/" + doc.get("filename", "")),
        "transcript": full_text,
        "analysis": analysis or "",
        "chunk_count": doc.get("chunk_count", 0),
        "summary": doc.get("summary", "")
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




# ─── SM-2 间隔重复复习系统 ───

def load_review_items():
    path = os.path.join(INDEX_DIR, "review_items.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_review_items(items):
    with open(os.path.join(INDEX_DIR, "review_items.json"), "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def sm2_update(item, quality):
    """SM-2 algorithm: update easiness factor, interval, repetitions."""
    ef = item.get("easiness_factor", 2.5)
    n = item.get("repetitions", 0)
    interval = item.get("interval", 0)

    if quality < 3:
        n = 0
        interval = 1
    else:
        if n == 0:
            interval = 1
        elif n == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        n += 1

    ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ef = max(1.3, ef)

    item["easiness_factor"] = ef
    item["interval"] = interval
    item["repetitions"] = n
    item["next_review"] = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                           .strftime("%Y-%m-%d") + f" +{interval}d")
    item["last_reviewed"] = datetime.now().isoformat()

    history = item.get("history", [])
    history.append({"date": datetime.now().isoformat(), "quality": quality})
    item["history"] = history[-20:]
    return item

def auto_generate_review_items(doc_id, weak_concepts, doc_title="", context=""):
    """从薄弱概念自动生成复习卡片"""
    items = load_review_items()
    existing_ids = {it["id"] for it in items}
    added = []

    for concept in weak_concepts:
        cid = hashlib.md5(f"{doc_id}:{concept}".encode()).hexdigest()[:12]
        if cid in existing_ids:
            continue
        items.append({
            "id": cid,
            "concept": concept,
            "source_doc": doc_title,
            "source_doc_id": doc_id,
            "context": context[:500] if context else "",
            "easiness_factor": 2.5,
            "interval": 0,
            "repetitions": 0,
            "next_review": datetime.now().strftime("%Y-%m-%d"),
            "created_at": datetime.now().isoformat(),
            "last_reviewed": None,
            "history": [],
            "type": "concept",
            "source": "feynman"
        })
        existing_ids.add(cid)
        added.append(cid)

    if added:
        save_review_items(items)
    return added

def auto_generate_exam_review(doc_id, errors, doc_title=""):
    """从错题自动生成复习卡片"""
    items = load_review_items()
    existing_ids = {it["id"] for it in items}
    added = []

    for err in errors:
        if isinstance(err, dict):
            question = err.get("question", err.get("raw", str(err)))
            answer = err.get("answer", "")
        else:
            question = str(err)
            answer = ""

        cid = hashlib.md5(f"{doc_id}:exam:{question}".encode()).hexdigest()[:12]
        if cid in existing_ids:
            continue
        items.append({
            "id": cid,
            "concept": question[:100],
            "source_doc": doc_title,
            "source_doc_id": doc_id,
            "context": f"正确答案: {answer}" if answer else "",
            "easiness_factor": 2.5,
            "interval": 0,
            "repetitions": 0,
            "next_review": datetime.now().strftime("%Y-%m-%d"),
            "created_at": datetime.now().isoformat(),
            "last_reviewed": None,
            "history": [],
            "type": "error",
            "source": "exam"
        })
        existing_ids.add(cid)
        added.append(cid)

    if added:
        save_review_items(items)
    return added

def get_due_reviews():
    """Get review items due today or overdue."""
    items = load_review_items()
    today_str = datetime.now().strftime("%Y-%m-%d")
    due = []

    for item in items:
        nr = item.get("next_review", "")
        # Parse: "2026-05-28" or "2026-05-28 +6d"
        review_date = nr.split(" +")[0] if " +" in nr else nr
        if not review_date:
            review_date = item.get("created_at", today_str)[:10]

        try:
            rd = datetime.strptime(review_date, "%Y-%m-%d")
            if rd <= datetime.now():
                due.append(item)
        except:
            due.append(item)

    due.sort(key=lambda x: x.get("next_review", "9999"))
    return due

def get_review_stats():
    """Get review statistics."""
    items = load_review_items()
    due = get_due_reviews()
    total = len(items)
    reviewed = sum(1 for it in items if it.get("last_reviewed"))
    mastered = sum(1 for it in items if it.get("repetitions", 0) >= 3 and it.get("easiness_factor", 2.5) >= 2.3)

    # Streak (consecutive days with reviews)
    streak = 0
    all_history = []
    for it in items:
        all_history.extend(it.get("history", []))
    all_history.sort(key=lambda x: x["date"], reverse=True)
    seen_dates = set()
    today = datetime.now().date()
    for i in range(30):
        d = (today - __import__("datetime").timedelta(days=i)).isoformat()[:10]
        if any(h["date"][:10] == d for h in all_history):
            streak += 1
        elif i > 0:
            break

    return {
        "total": total,
        "due_today": len(due),
        "reviewed": reviewed,
        "mastered": mastered,
        "streak": streak
    }


@app.route("/api/review", methods=["GET", "POST"])
def review():
    """SM-2 间隔重复复习：GET 获取今日待复习，POST 提交复习结果"""
    if request.method == "GET":
        page = request.args.get("page", "1")
        per_page = int(request.args.get("per_page", "20"))

        if page == "stats":
            return jsonify(get_review_stats())

        due = get_due_reviews()

        # Paginate
        start = (int(page) - 1) * per_page
        page_items = due[start:start + per_page]

        # Remove large context for list view
        slim = []
        for it in page_items:
            s = dict(it)
            if len(s.get("context", "")) > 200:
                s["context"] = s["context"][:200] + "..."
            slim.append(s)

        return jsonify({
            "items": slim,
            "total_due": len(due),
            "page": int(page),
            "per_page": per_page,
            "stats": get_review_stats()
        })

    elif request.method == "POST":
        data = request.get_json()
        item_id = data.get("id", "")
        quality = data.get("quality", 0)
        doc_id = data.get("doc_id", "")
        concept = data.get("concept", "")
        context = data.get("context", "")

        # Action: auto-generate review items from weak concepts
        if data.get("action") == "generate":
            weak = data.get("weak_concepts", [])
            title = data.get("doc_title", "")
            added = auto_generate_review_items(doc_id, weak, title, context)
            return jsonify({"generated": len(added), "ids": added})

        # Action: generate from exam errors
        if data.get("action") == "generate_exam":
            errors = data.get("errors", [])
            title = data.get("doc_title", "")
            added = auto_generate_exam_review(doc_id, errors, title)
            return jsonify({"generated": len(added), "ids": added})

        # Standard: submit review quality
        if not item_id or quality < 1 or quality > 5:
            return jsonify({"error": "id and quality (1-5) required"}), 400

        items = load_review_items()
        updated = None
        for it in items:
            if it["id"] == item_id:
                sm2_update(it, quality)
                updated = it
                break

        if not updated:
            return jsonify({"error": "item not found"}), 404

        save_review_items(items)

        return jsonify({
            "ok": True,
            "item": updated,
            "stats": get_review_stats()
        })


@app.route("/api/review/<item_id>", methods=["GET"])
def review_detail(item_id):
    """Get a single review item with full context."""
    items = load_review_items()
    for it in items:
        if it["id"] == item_id:
            return jsonify(it)
    return jsonify({"error": "item not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)
