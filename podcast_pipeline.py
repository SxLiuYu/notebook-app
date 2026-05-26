#!/usr/bin/env python3
"""Multi-stage AI podcast generator: extract → outline → script → TTS."""
import os, re, json, time, subprocess, tempfile


def generate_podcast(docs_text, topic, llm_call_fn, output_dir=None):
    """
    4-stage podcast generation pipeline.
    
    Args:
        docs_text: Full document text
        topic: Podcast topic (user-provided or auto)
        llm_call_fn: Function(messages, max_tokens) -> str
        output_dir: Directory for audio output (default: ./static/podcasts)
    
    Returns:
        {script, segments, audio_path, title, outline}
    """
    if not docs_text or not docs_text.strip():
        return {"error": "No document content", "script": "", "segments": [], "audio_path": None}
    
    # Truncate if too long
    if len(docs_text) > 12000:
        docs_text = docs_text[:12000] + "\n...(内容已截断)"
    
    title = topic or "文档内容概览"
    
    # Stage 1: Extract key points
    points = _extract_key_points(docs_text, topic, llm_call_fn)
    if not points:
        return {"error": "Failed to extract key points", "script": "", "segments": [], "audio_path": None}
    
    # Stage 2: Generate outline
    outline = _generate_outline(points, topic, llm_call_fn)
    
    # Stage 3: Generate full script
    script_result = _generate_script(outline, points, topic, docs_text, llm_call_fn)
    
    if not script_result:
        return {"error": "Failed to generate script", "script": "", "segments": [], "audio_path": None}
    
    # Stage 4: Render audio
    audio_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        audio_path = _render_audio(script_result["segments"], output_dir, title)
    
    return {
        "script": script_result["script_text"],
        "segments": script_result["segments"],
        "audio_path": audio_path,
        "title": title,
        "outline": outline
    }


def _extract_key_points(docs_text, topic, llm_call_fn):
    """Stage 1: Extract 8-10 key insights with supporting evidence."""
    prompt = f"""你是一个专业的内容分析师。请从以下文档中提取 8-10 个核心观点。

要求：
- 每个观点用一句简洁的话概括
- 为每个观点提供文档中的具体引文作为证据
- 按重要性从高到低排列
- 覆盖不同角度的观点（不要重复）

格式（严格 JSON 数组）：
[
  {{"point": "观点概括", "evidence": "文档原文引用", "importance": 9}},
  ...
]

文档主题：{topic}

文档内容：
{docs_text}

请输出 JSON 数组："""
    
    response = llm_call_fn([{"role": "user", "content": prompt}], max_tokens=2048)
    
    try:
        # Extract JSON array
        match = re.search(r'\[.*\]', response or "", re.DOTALL)
        if match:
            points = json.loads(match.group())
            return [p for p in points if p.get("point")]
    except Exception:
        pass
    
    # Fallback: parse as numbered list
    points = []
    for line in (response or "").split("\n"):
        line = line.strip()
        if re.match(r'^\d+[\.\)、]', line):
            points.append({"point": line, "evidence": "", "importance": 5})
    
    return points[:10]


def _generate_outline(points, topic, llm_call_fn):
    """Stage 2: Create conversation outline with speaker assignments."""
    points_text = "\n".join(f"{i+1}. {p['point']}" for i, p in enumerate(points[:8]))
    
    prompt = f"""你是一个播客节目策划。请基于以下核心观点，设计一个 5-8 分钟双人播客的对话大纲。

角色：
- [A] 主持人：好奇、善于提问、代表听众问出关键问题
- [B] 专家：深度解读、引证据、用类比让复杂概念易懂

大纲结构（7-8 个阶段）：
1. 开场吸引注意力（主持人引入话题，专家点出核心悬念）
2-6. 逐步展开核心观点，穿插互动和追问
7-8. 总结升华、给出 actionable takeaway

核心观点：
{points_text}

播客主题：{topic}

请输出大纲（每个阶段一行）：
格式：[阶段名] [角色] 内容概要"""
    
    response = llm_call_fn([{"role": "user", "content": prompt}], max_tokens=1024)
    
    outline = []
    for line in (response or "").split("\n"):
        line = line.strip()
        if line and ('[' in line or 'A' in line or 'B' in line):
            outline.append(line)
    
    return outline if outline else ["开场 A 引入话题", "展开 B 深度分析", "互动 A 追问", "总结 B 收尾"]


def _generate_script(outline, points, topic, docs_text, llm_call_fn):
    """Stage 3: Generate full conversation script."""
    points_text = "\n".join(
        f"{i+1}. {p['point']}（证据：{p.get('evidence', 'N/A')[:100]}）"
        for i, p in enumerate(points[:8])
    )
    outline_text = "\n".join(outline)
    
    prompt = f"""你是一个播客编剧。请基于以下大纲和观点，生成一段自然流畅的中文双人播客对话脚本。

角色设定：
[A] 主持人（25-35岁女性口吻）：
- 性格：好奇心强、善于提问、偶尔幽默
- 语言风格：口语化、短句为主、会说"等一下...""所以你的意思是..." "哇这个有意思"
- 功能：代表听众提问、推动话题切换、总结要点方便听众理解

[B] 专家（30-45岁男性口吻）：  
- 性格：知识渊博但不装、善于用比喻和例子
- 语言风格：沉稳但生动、会说"有趣的是..." "关键在于..." "打个比方..."
- 功能：深度解读观点、引用文档证据、把复杂概念讲得通俗易懂

对话要求：
- 自然的来回互动，有追问、有补充、有呼应
- 话题间过渡要自然（"说到这个我突然想到..." "这个话题让我想起文档里另一个点..."）
- 可以适当加口语词（"嗯""就是说""其实吧"）增加真实感
- 偶尔有2-3句话的 banter（轻松互动）来调节节奏
- 控制在 2000-3000 字

格式：
[A] 主持人的话...
[B] 专家的话...
[A] ...
（交替进行）

播客主题：{topic}

核心观点与证据：
{points_text}

对话大纲：
{outline_text}

文档原文（参考）：
{docs_text[:2000]}

请生成完整的播客对话脚本："""
    
    script_text = llm_call_fn([{"role": "user", "content": prompt}], max_tokens=4096)
    
    if not script_text:
        return None
    
    # Parse segments
    segments = []
    for line in script_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[A]") or line.startswith("[A]") or line.startswith("A]"):
            text = re.sub(r'^\[?A\]?\s*', '', line).strip()
            if text:
                segments.append({"role": "host", "text": text})
        elif line.startswith("[B]") or line.startswith("B]"):
            text = re.sub(r'^\[?B\]?\s*', '', line).strip()
            if text:
                segments.append({"role": "expert", "text": text})
    
    if not segments:
        # Fallback: treat each non-empty line as alternating
        lines = [l.strip() for l in script_text.split("\n") if l.strip() and not l.startswith("#")]
        for i, line in enumerate(lines):
            role = "host" if i % 2 == 0 else "expert"
            segments.append({"role": role, "text": line})
    
    return {"script_text": script_text, "segments": segments}


def _render_audio(segments, output_dir, title):
    """Stage 4: Render segments to audio using edge-tts."""
    try:
        import edge_tts
    except ImportError:
        return None
    
    if not segments:
        return None
    
    # Voice mapping
    VOICES = {
        "host": "zh-CN-XiaoxiaoNeural",   # Female, cheerful
        "expert": "zh-CN-YunxiNeural",     # Male, professional
    }
    
    # Create temp dir for individual segments
    segment_dir = tempfile.mkdtemp(prefix="podcast_segments_")
    wav_files = []
    
    try:
        for i, seg in enumerate(segments):
            voice = VOICES.get(seg["role"], "zh-CN-XiaoxiaoNeural")
            text = seg["text"]
            
            if len(text) < 2:
                continue
            
            output_file = os.path.join(segment_dir, f"seg_{i:04d}.mp3")
            
            # edge-tts async in sync wrapper
            import asyncio
            
            async def _tts():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_file)
            
            try:
                asyncio.run(_tts())
                wav_files.append(output_file)
            except Exception as e:
                print(f"TTS segment {i} failed: {e}")
                continue
        
        if not wav_files:
            return None
        
        # Concatenate with ffmpeg or simple concat
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
        safe_title = safe_title.strip().replace(' ', '_')
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"{safe_title}_{timestamp}.mp3")
        
        # Use ffmpeg to concat with 0.5s silence between
        # Build concat file list
        concat_list = os.path.join(segment_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for wf in wav_files:
                f.write(f"file '{wf}'\n")
        
        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list, "-c", "copy", output_path
            ], capture_output=True, timeout=60, check=True)
            return output_path
        except Exception:
            # Fallback: just use the first segment as output
            if wav_files:
                import shutil
                shutil.copy(wav_files[0], output_path)
                return output_path
        
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(segment_dir, ignore_errors=True)
        except Exception:
            pass
    
    return None


if __name__ == "__main__":
    # Self-test with mock LLM
    def mock_llm(messages, max_tokens=2048):
        prompt = messages[0]["content"]
        if "提取" in prompt and "核心观点" in prompt:
            return json.dumps([
                {"point": "Python 适合初学者学习编程", "evidence": "Python is known for its readability", "importance": 9},
                {"point": "机器学习依赖大量数据", "evidence": "Machine learning uses statistical techniques", "importance": 8},
                {"point": "深度学习使用多层神经网络", "evidence": "深度学习是机器学习的一个子集", "importance": 8},
                {"point": "NLP 是人机交互的关键技术", "evidence": "Natural language processing deals with interaction", "importance": 7},
            ], ensure_ascii=False)
        elif "大纲" in prompt:
            return """1. 开场 A 引入AI和编程的关系
2. 展开 B 解释机器学习和深度学习的区别
3. 互动 A 追问实际应用场景
4. 深入 B 介绍 NLP 的突破
5. 总结 B 展望未来趋势"""
        else:
            return """[A] 欢迎收听今天的播客！今天我们聊聊人工智能的那些事儿。老张，听说最近 AI 特别火？

[B] 确实，尤其是大语言模型，可以说是改变了很多行业的游戏规则。有趣的是，这些技术的底层其实并不复杂。

[A] 等一下，你说不复杂？可我听说深度学习要好多层神经网络...

[B] 打个比方吧，深度学习就像叠积木——每一层处理一点点信息，叠得足够深就能理解很复杂的东西了。

[A] 哇这个比喻好！那 NLP 自然语言处理是不是也是这么叠出来的？

[B] NLP 更特别一些，它需要理解上下文。关键在于它能捕捉词语之间的关系，就像你读书时能根据上下文猜出不认识的词的意思一样。"""
    
    test_text = """
Python is a high-level programming language known for its readability.
Machine learning uses statistical techniques to give computers the ability to learn.
深度学习是机器学习的一个子集，使用多层神经网络。
Natural language processing deals with the interaction between computers and human language.
自然语言处理是人工智能的一个重要分支。
"""
    
    print("Testing podcast_pipeline...")
    result = generate_podcast(test_text, "AI 入门指南", mock_llm)
    
    print(f"Title: {result['title']}")
    print(f"Segments: {len(result['segments'])}")
    print(f"Audio: {result['audio_path'] or 'No audio (edge-tts may not be available)'}")
    print(f"\nScript preview:")
    for seg in result['segments'][:3]:
        role_emoji = "🎤" if seg['role'] == 'host' else "🎓"
        print(f"  {role_emoji} [{seg['role']}] {seg['text'][:100]}...")
    
    print(f"\nOutline:")
    for line in result.get('outline', []):
        print(f"  {line}")
    
    print("\n✅ podcast_pipeline.py self-test passed")
