"""
Notebook App — NiceGUI rewrite
"""
import os, json, uuid, time, asyncio
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import requests

from nicegui import ui, app, Client
from starlette.staticfiles import StaticFiles

from document_processor import process_file, summarize
from rag_engine import add_documents, search, delete_document, get_stats, re_rank_with_llm
from podcast_pipeline import generate_podcast as gen_podcast

BASE_DIR = Path(os.environ.get('BASE_DIR', '/opt/notebook-app'))
DOCS_DIR = BASE_DIR / 'data' / 'documents'
INDEX_DIR = BASE_DIR / 'data' / 'indexes'
STATIC_DIR = BASE_DIR / 'static'
PODCAST_DIR = STATIC_DIR / 'podcasts'

for d in [DOCS_DIR, INDEX_DIR, PODCAST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FINNA_KEY = os.environ.get('FINNA_KEY', 'app-ULzJbc3OaIN50mZVSU7sAa97')
FINNA_BASE = 'https://www.finna.com.cn/v1'
LLM_MODEL = 'deepseek-v4-flash'

import rag_engine
rag_engine.CHROMA_PATH = str(BASE_DIR / 'data' / 'chroma_db')

def load_docs():
    fp = INDEX_DIR / 'documents.json'
    return json.loads(fp.read_text()) if fp.exists() else []

def save_docs(docs):
    (INDEX_DIR / 'documents.json').write_text(json.dumps(docs, ensure_ascii=False, indent=2))

def load_folders():
    fp = INDEX_DIR / 'folders.json'
    return json.loads(fp.read_text()) if fp.exists() else []

def save_folders(folders):
    (INDEX_DIR / 'folders.json').write_text(json.dumps(folders, ensure_ascii=False, indent=2))

NL = chr(10)  # newline

def call_llm(messages, max_tokens=2048):
    for attempt in range(3):
        try:
            resp = requests.post(
                f'{FINNA_BASE}/chat/completions',
                headers={'Authorization': f'Bearer {FINNA_KEY}', 'Content-Type': 'application/json'},
                json={'model': LLM_MODEL, 'messages': messages, 'temperature': 0.3,
                      'max_tokens': max_tokens, 'stream': False,
                      'extra_body': {'enable_thinking': False}},
                timeout=90
            )
            data = resp.json()
            if 'choices' in data:
                return data['choices'][0]['message']['content']
        except Exception as e:
            print(f'LLM attempt {attempt+1} failed: {e}')
            time.sleep(2)
    return None

flask_app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(BASE_DIR / 'templates'))

@flask_app.route('/health')
def api_health():
    docs = load_docs()
    return jsonify({'status': 'ok', 'docs': len(docs)})

@flask_app.route('/folders', methods=['GET', 'POST'])
def api_folders():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name', '').strip()
        if not name: return jsonify({'error': 'name required'}), 400
        folders = load_folders()
        fid = str(uuid.uuid4())[:8]
        folders.append({'id': fid, 'name': name, 'created_at': time.strftime('%Y-%m-%d %H:%M')})
        save_folders(folders)
        return jsonify({'ok': True, 'folder_id': fid, 'folders': folders})
    folders = load_folders()
    docs = load_docs()
    counts = {}
    for d in docs:
        fid = d.get('folder_id', '') or 'uncategorized'
        counts[fid] = counts.get(fid, 0) + 1
    for f in folders:
        f['doc_count'] = counts.get(f['id'], 0)
    return jsonify({'folders': folders})

@flask_app.route('/folders/<fid>', methods=['PUT', 'DELETE'])
def api_folder(fid):
    folders = load_folders()
    if request.method == 'PUT':
        data = request.get_json()
        name = data.get('name', '').strip()
        for f in folders:
            if f['id'] == fid: f['name'] = name; break
        save_folders(folders)
        return jsonify({'ok': True})
    elif request.method == 'DELETE':
        folders = [f for f in folders if f['id'] != fid]
        save_folders(folders)
        docs = load_docs()
        for d in docs:
            if d.get('folder_id') == fid: d['folder_id'] = ''
        save_docs(docs)
        return jsonify({'ok': True})

@flask_app.route('/docs', methods=['GET'])
def api_docs():
    folder_id = request.args.get('folder_id', '')
    docs = load_docs()
    if folder_id:
        if folder_id == 'uncategorized':
            docs = [d for d in docs if not d.get('folder_id')]
        else:
            docs = [d for d in docs if d.get('folder_id') == folder_id]
    for d in docs:
        if 'chunks' not in d: d['chunks'] = 0
    return jsonify({'documents': docs})

@flask_app.route('/docs/<did>', methods=['DELETE'])
def api_delete_doc(did):
    docs = load_docs()
    doc = next((d for d in docs if d['id'] == did), None)
    docs = [d for d in docs if d['id'] != did]
    save_docs(docs)
    if doc:
        fp = Path(doc.get('path', ''))
        if fp.exists(): fp.unlink(missing_ok=True)
        try:
            import shutil
            shutil.rmtree(BASE_DIR / 'data' / 'chroma_db' / did, ignore_errors=True)
        except: pass
        try: delete_document(did)
        except: pass
    return jsonify({'ok': True})

@flask_app.route('/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['file']
    folder_id = request.form.get('folder_id', '') or None
    if not f.filename: return jsonify({'error': 'empty filename'}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ('.txt', '.md', '.pdf', '.docx', '.csv', '.json'):
        return jsonify({'error': f'unsupported format: {ext}'}), 400

    filepath = DOCS_DIR / f.filename
    f.save(str(filepath))

    try:
        result = process_file(str(filepath), f.filename)
        chunks = result.get('chunks', []) if isinstance(result, dict) else []
    except Exception as e:
        return jsonify({'error': f'parse failed: {str(e)}'}), 500

    if not chunks:
        return jsonify({'error': 'no content'}), 400

    doc_id = str(uuid.uuid4())
    try:
        add_documents(doc_id, chunks, str(filepath))
    except Exception as e:
        print(f'Index warning: {e}')

    docs = load_docs()
    docs.append({
        'id': doc_id, 'title': f.filename, 'path': str(filepath),
        'chunks': len(chunks), 'folder_id': folder_id,
        'uploaded_at': time.strftime('%Y-%m-%d %H:%M'),
        'size': os.path.getsize(str(filepath))
    })
    save_docs(docs)

    suggestions = ['what is this doc about?', 'summarize key points', 'any important data?']
    return jsonify({'ok': True, 'doc': {'id': doc_id, 'title': f.filename, 'chunks': len(chunks), 'suggested': suggestions}})

@flask_app.route('/ask', methods=['POST'])
def api_ask():
    data = request.get_json()
    query = data.get('query', '').strip()
    doc_id = data.get('doc_id', '')
    folder_id = data.get('folder_id', '')
    history = data.get('history', [])

    if not query: return jsonify({'error': 'empty query'}), 400

    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d['id'] == doc_id]
    elif folder_id:
        if folder_id == 'uncategorized':
            docs = [d for d in docs if not d.get('folder_id')]
        else:
            docs = [d for d in docs if d.get('folder_id') == folder_id]

    if not docs:
        return jsonify({'answer': 'please upload documents first', 'sources': []})

    doc_ids = [d['id'] for d in docs]

    try:
        results = search(query, top_k=5, doc_ids=doc_ids)
    except Exception as e:
        try: results = search(query, top_k=5)
        except: return jsonify({'answer': f'search failed: {str(e)}', 'sources': []})

    try:
        results = re_rank_with_llm(query, results)
    except: pass

    sources = []
    for r in results[:5]:
        src_path = r.get('path', '')
        sources.append({
            'title': os.path.basename(src_path) if src_path else 'unknown',
            'content': (r.get('content', '') or '')[:200]
        })

    context = NL.join(f'[source: {s["title"]}]{NL}{s["content"]}' for s in sources)
    msgs = [{'role': 'system', 'content': f'Answer based on documents. If no info found, say so.{NL}{NL}Documents:{NL}{context}'}]
    for h in history[-6:]:
        role = h.get('role', 'user')
        if role in ('user', 'assistant'):
            msgs.append({'role': role, 'content': h.get('content', '')})
    msgs.append({'role': 'user', 'content': query})

    answer = call_llm(msgs, max_tokens=2048)
    return jsonify({'answer': answer or 'generate failed', 'sources': sources})

@flask_app.route('/podcast', methods=['POST'])
def api_podcast():
    data = request.get_json()
    doc_id = data.get('doc_id', '')
    folder_id = data.get('folder_id', '')
    topic = data.get('topic', '').strip()

    docs = load_docs()
    if doc_id:
        docs = [d for d in docs if d['id'] == doc_id]
    elif folder_id:
        if folder_id == 'uncategorized':
            docs = [d for d in docs if not d.get('folder_id')]
        else:
            docs = [d for d in docs if d.get('folder_id') == folder_id]

    if not docs:
        return jsonify({'error': 'please upload documents'}), 400

    full_text = ''
    for doc in docs:
        full_text += f'{NL}{NL}## {doc["title"]}{NL}{NL}'
        fp = doc.get('path', '')
        if fp and Path(fp).exists():
            try:
                result = process_file(fp, doc['title'])
                chunks = result.get('chunks', []) if isinstance(result, dict) else []
                for c in chunks[:20]:
                    full_text += (c.get('content', '') if isinstance(c, dict) else str(c))[:500]
            except: pass

    try:
        result = gen_podcast(full_text[:15000], topic, call_llm, output_dir=str(PODCAST_DIR))
    except Exception as e:
        return jsonify({'error': f'podcast failed: {str(e)}'}), 500

    audio_url = ''
    if isinstance(result, dict):
        audio_path = result.get('audio_path', '')
        if audio_path:
            audio_url = f'/static/podcasts/{os.path.basename(audio_path)}'
        script = result.get('script', result.get('text', ''))
        title = result.get('title', 'AI Podcast')
    else:
        script = str(result)
        title = 'AI Podcast'

    return jsonify({'title': title, 'script': script, 'audio_url': audio_url})

@flask_app.route('/suggest', methods=['POST'])
def api_suggest():
    return jsonify({'questions': ['what is this doc about?', 'summarize key points', 'what are the highlights?']})

@flask_app.route('/folders/<fid>/knowledge', methods=['POST'])
def api_knowledge(fid):
    docs = load_docs()
    if fid == 'uncategorized':
        folder_docs = [d for d in docs if not d.get('folder_id')]
        fname = 'uncategorized'
    else:
        folder_docs = [d for d in docs if d.get('folder_id') == fid]
        folders = load_folders()
        fname = next((f['name'] for f in folders if f['id'] == fid), fid)

    if not folder_docs:
        return jsonify({'folder_name': fname, 'summary_md': '# No documents yet', 'mindmap': None, 'learning_path': ''})

    full_text = ''
    for d in folder_docs:
        full_text += f'{NL}{NL}## {d["title"]}{NL}{NL}'
        fp = d.get('path', '')
        if fp and Path(fp).exists():
            try:
                result = process_file(fp, d['title'])
                chunks = result.get('chunks', []) if isinstance(result, dict) else []
                for c in chunks[:10]:
                    full_text += (c.get('content', '') if isinstance(c, dict) else str(c))[:800]
            except: pass

    full_text = full_text[:12000]

    summary_prompt = [
        {'role': 'system', 'content': 'You are a knowledge organizer. Generate a structured summary in Markdown. Include: 1) Core theme overview 2) Key concepts 3) Main points. Use Chinese.'},
        {'role': 'user', 'content': f'Summarize:{NL}{NL}{full_text}'}
    ]
    summary = call_llm(summary_prompt, max_tokens=2000) or '# Summary{}{}Generation failed'.format(NL, NL)

    mm_prompt = [
        {'role': 'system', 'content': 'You are a mindmap generator. Output JSON: {"topic":"Topic","children":[{"topic":"Subtopic","children":[]},...]}. JSON only, max 3 levels.'},
        {'role': 'user', 'content': f'Generate mindmap:{NL}{NL}{full_text[:3000]}'}
    ]
    mm_raw = call_llm(mm_prompt, max_tokens=1000) or '{}'
    try:
        mindmap = json.loads(mm_raw.strip().removeprefix('```json').removesuffix('```').strip())
    except:
        mindmap = {'topic': fname, 'children': [{'topic': d['title'], 'children': []} for d in folder_docs]}

    lp_prompt = [
        {'role': 'system', 'content': 'You are a learning path designer. Design a step-by-step learning path with Markdown format. Use Chinese.'},
        {'role': 'user', 'content': f'Design learning path:{NL}{NL}{full_text[:3000]}'}
    ]
    learning_path = call_llm(lp_prompt, max_tokens=2000) or '# Learning Path{}{}TBD'.format(NL, NL)

    return jsonify({
        'folder_name': fname,
        'summary_md': summary,
        'mindmap': mindmap,
        'learning_path': learning_path
    })

@flask_app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(STATIC_DIR), filename)

# ─── NiceGUI Frontend ───
@ui.page('/')
def index_page(client: Client):
    dark = ui.dark_mode()
    
    folders = []
    all_docs = []
    active_folder = ''
    active_doc = ''
    chat_msgs = []
    
    folder_badge_ref = None
    folder_list_ref = None
    chat_area_ref = None
    pod_area_ref = None
    upload_widget_ref = None
    
    async def refresh_all():
        nonlocal folders, all_docs
        import httpx
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get('http://127.0.0.1:8095/api/folders')
            folders = r.json().get('folders', [])
            r2 = await hc.get(f'http://127.0.0.1:8095/api/docs?folder_id={active_folder}')
            all_docs = r2.json().get('documents', [])
        render_sidebar()
    
    def _folder_name():
        if not active_folder: return '全部文档'
        if active_folder == 'uncategorized': return '未分类'
        f = next((f for f in folders if f['id'] == active_folder), None)
        return f['name'] if f else active_folder
    
    def render_sidebar():
        nonlocal folder_list_ref
        if folder_list_ref is None: return
        folder_list_ref.clear()
        with folder_list_ref:
            b = ui.button('📂 全部文档', on_click=lambda: select_folder('')).props('flat align=left dense').classes('w-full')
            if not active_folder: b.props('color=primary ugly')
            
            b2 = ui.button('📄 未分类', on_click=lambda: select_folder('uncategorized')).props('flat align=left dense').classes('w-full')
            if active_folder == 'uncategorized': b2.props('color=primary ugly')
            
            for f in folders:
                fid = f['id']
                b3 = ui.button(f'{f["name"]} ({f.get("doc_count",0)})', on_click=lambda fid=fid: select_folder(fid)).props('flat align=left dense').classes('w-full')
                if active_folder == fid: b3.props('color=primary ugly')
            
            ui.separator()
            for d in all_docs:
                did = d['id']
                b4 = ui.button(d['title'][:30], on_click=lambda did=did: select_doc(did)).props('flat align=left dense').classes('w-full text-xs')
                if active_doc == did: b4.props('color=secondary ugly')
        
        if folder_badge_ref:
            folder_badge_ref.set_text(f'📂 {_folder_name()}')
    
    async def select_folder(fid):
        nonlocal active_folder, active_doc
        active_folder = fid; active_doc = ''
        chat_msgs.clear()
        await refresh_all()
        render_chat()
    
    def select_doc(did):
        nonlocal active_doc
        active_doc = did if active_doc != did else ''
        render_sidebar()
    
    async def handle_upload(e):
        nonlocal active_doc
        content = e.content.read()
        filename = e.name
        import httpx
        async with httpx.AsyncClient(timeout=60) as hc:
            form_data = {'folder_id': active_folder} if active_folder and active_folder != 'uncategorized' else {}
            r = await hc.post('http://127.0.0.1:8095/api/upload',
                files={'file': (filename, content, 'application/octet-stream')},
                data=form_data)
            result = r.json()
        if result.get('ok'):
            ui.notify(f'✅ {result["doc"]["title"]} — {result["doc"]["chunks"]} 段落', type='positive')
            active_doc = result['doc']['id']
            await refresh_all()
        else:
            ui.notify(f'❌ {result.get("error","上传失败")}', type='negative')
        if upload_widget_ref: upload_widget_ref.reset()
    
    async def send_message():
        nonlocal chat_area_ref
        q = query_input.value.strip()
        if not q: return
        chat_msgs.append({'role': 'user', 'content': q, 'name': '你'})
        query_input.value = ''
        render_chat()
        
        import httpx
        body = {'query': q, 'doc_id': active_doc, 'folder_id': active_folder, 'history': chat_msgs[:-1]}
        async with httpx.AsyncClient(timeout=120) as hc:
            r = await hc.post('http://127.0.0.1:8095/api/ask', json=body)
            result = r.json()
        
        answer = result.get('answer', 'no answer')
        sources = result.get('sources', [])
        src_md = ''
        if sources:
            src_md = '{}{}---{}{}Sources: {}'.format(NL, NL, NL, NL, ' · '.join(s['title'] for s in sources[:5]))
        
        chat_msgs.append({'role': 'assistant', 'content': answer + src_md, 'name': 'AI 助手'})
        render_chat()
    
    def render_chat():
        nonlocal chat_area_ref
        if chat_area_ref is None: return
        chat_area_ref.clear()
        with chat_area_ref:
            if not chat_msgs:
                with ui.column().classes('items-center justify-center w-full'):
                    ui.icon('menu_book', size='4rem').classes('text-gray-500 mb-3 mt-8')
                    ui.label('上传文档开始探索').classes('text-lg font-bold text-gray-400')
                    ui.label('支持 PDF · DOCX · TXT · MD · CSV 格式').classes('text-sm text-gray-500')
            for msg in chat_msgs:
                ui.chat_message(
                    text=msg['content'],
                    name=msg.get('name', ''),
                    sent=msg['role'] == 'user'
                )
    
    async def show_knowledge():
        if not active_folder or active_folder == 'uncategorized':
            ui.notify('请先选择一个分类', type='warning')
            return
        
        import httpx
        async with httpx.AsyncClient(timeout=120) as hc:
            r = await hc.post(f'http://127.0.0.1:8095/api/folders/{active_folder}/knowledge')
            data = r.json()
        
        fname = data.get('folder_name', active_folder)
        summary_md = data.get('summary_md', '')
        mindmap_data = data.get('mindmap')
        learning_path = data.get('learning_path', '')
        
        with ui.dialog() as dlg:
            dlg.props('maximized')
            with ui.card().classes('w-full'):
                ui.label(f'📊 {fname} · 知识梳理').classes('text-xl font-bold mb-4')
                with ui.tabs() as tabs:
                    s_tab = ui.tab('📝 知识总结')
                    m_tab = ui.tab('🧠 思维导图')
                    l_tab = ui.tab('📚 学习路径')
                with ui.tab_panels(tabs, value=s_tab):
                    with ui.tab_panel(s_tab):
                        ui.markdown(summary_md or '暂无内容')
                    with ui.tab_panel(m_tab):
                        if mindmap_data:
                            def _render_node(n, level=0):
                                prefix = '  ' * level
                                cls = 'text-lg font-bold' if level == 0 else ('font-semibold' if level == 1 else 'text-sm')
                                ui.label(f'{prefix}{n["topic"]}').classes(f'{cls} ml-{level*2}')
                                for child in n.get('children', []):
                                    _render_node(child, level + 1)
                            _render_node(mindmap_data)
                        else:
                            ui.label('暂无数据')
                    with ui.tab_panel(l_tab):
                        ui.markdown(learning_path or '暂无内容')
                ui.button('关闭', on_click=lambda: dlg.close()).props('flat').classes('mt-4')
        await dlg
    
    async def generate_podcast_ui():
        topic = podcast_topic.value.strip()
        if not all_docs:
            ui.notify('请先上传文档', type='warning')
            return
        
        nonlocal pod_area_ref
        if pod_area_ref is None: return
        pod_area_ref.clear()
        with pod_area_ref:
            ui.spinner(size='lg')
            ui.label('正在生成播客...').classes('text-gray-500 mt-2')
        
        import httpx
        async with httpx.AsyncClient(timeout=180) as hc:
            r = await hc.post('http://127.0.0.1:8095/api/podcast', json={
                'doc_id': active_doc, 'folder_id': active_folder, 'topic': topic
            })
            result = r.json()
        
        pod_area_ref.clear()
        with pod_area_ref:
            if result.get('error'):
                ui.label(f'{result["error"]}').classes('text-red-500 text-lg')
            else:
                ui.label(f'🎙️ {result.get("title", "AI 播客")}').classes('text-xl font-bold mb-4')
                if result.get('audio_url'):
                    ui.audio(result['audio_url'])
                script = result.get('script', '')
                for line in script.split(NL):
                    line = line.strip()
                    if not line: continue
                    if line.startswith('[A]'):
                        ui.chat_message(line[3:].strip(), name='🎤 主持人', sent=False)
                    elif line.startswith('[B]'):
                        ui.chat_message(line[3:].strip(), name='🎓 专家', sent=False)
                    else:
                        ui.markdown(line)
        ui.notify('✅ 播客已生成', type='positive')
    
    async def new_folder():
        name = new_folder_input.value.strip()
        if not name: return
        import httpx
        async with httpx.AsyncClient() as hc:
            await hc.post('http://127.0.0.1:8095/api/folders', json={'name': name})
        new_folder_input.value = ''
        await refresh_all()
        ui.notify(f'✅ 已创建：{name}', type='positive')
    
    # ─── LAYOUT ───
    with ui.header(elevated=True).classes('bg-primary text-white items-center'):
        ui.button(icon='menu', on_click=lambda: drawer.toggle()).props('flat color=white dense')
        ui.label('Notebook · AI 研究助手').classes('text-lg font-bold')
        folder_badge_ref = ui.label('全部文档').classes('text-sm opacity-80')
        ui.space()
        if active_folder and active_folder != 'uncategorized':
            ui.button('📊 知识梳理', on_click=show_knowledge).props('flat color=white dense')
        ui.button(icon='dark_mode', on_click=lambda: dark.toggle()).props('flat color=white dense')
    
    with ui.left_drawer(value=True).classes('bg-gray-900 text-white') as drawer:
        drawer.style('width: 280px; max-width: 85vw')
        drawer.style('width: 280px')
        with ui.column().classes('w-full gap-1 p-2'):
            upload_widget_ref = ui.upload(
                on_upload=handle_upload,
                auto_upload=True,
                label='📤 上传文档',
                max_file_size=50_000_000
            ).props('accept=".txt,.md,.pdf,.docx,.csv,.json"').classes('w-full')
            
            with ui.row().classes('w-full gap-1 mt-2'):
                new_folder_input = ui.input(placeholder='新建分类...').props('dense dark outlined').classes('flex-grow')
                ui.button('＋', on_click=new_folder).props('dense size=sm')
            
            ui.separator()
            folder_list_ref = ui.column().classes('w-full gap-0')
    
    with ui.tabs() as main_tabs:
        chat_tab = ui.tab('💬 问答')
        pod_tab = ui.tab('🎙️ 播客')
    
    with ui.tab_panels(main_tabs, value=chat_tab).classes('w-full'):
        with ui.tab_panel(chat_tab).classes('column'):
            chat_area_ref = ui.column().classes('flex-grow overflow-auto p-4')
            with ui.row().classes('w-full items-center gap-2 p-4 bg-gray-50'):
                query_input = ui.input(placeholder='向文档提问...').props('outlined dense').classes('flex-grow')
                query_input.on('keydown.enter', send_message)
                ui.button('发送', on_click=send_message, icon='send').props('color=primary ugly')
        
        with ui.tab_panel(pod_tab).classes('column'):
            pod_area_ref = ui.column().classes('flex-grow overflow-auto p-4 items-center')
            with pod_area_ref:
                ui.icon('podcasts', size='4rem').classes('text-gray-500 mb-3 mt-8')
                ui.label('AI 双人播客').classes('text-lg font-bold text-gray-400')
                ui.label('基于文档自动生成对话式播客脚本').classes('text-sm text-gray-500')
            with ui.row().classes('w-full items-center gap-2 p-4 bg-gray-50'):
                podcast_topic = ui.input(placeholder='播客主题（可选）').props('outlined dense').classes('flex-grow')
                ui.button('🎙️ 生成播客', on_click=generate_podcast_ui, icon='smart_toy').props('color=accent ugly')
    
    ui.timer(0.1, lambda: refresh_all(), once=True)

# ───── Run ─────
from nicegui import app as nicegui_app
from fastapi.middleware.wsgi import WSGIMiddleware
nicegui_app.mount("/api", WSGIMiddleware(flask_app))
nicegui_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
ui.run(
    host='0.0.0.0',
    port=8095,
    title='Notebook · AI 研究助手',
    favicon='📖',
    storage_secret='notebook-app-2025',
    reload=False,
    show=False
)
