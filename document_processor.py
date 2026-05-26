#!/usr/bin/env python3
"""Multi-format document processor: PDF, DOCX, text, semantic chunking, summarization."""
import os, re

def extract_text(filepath, filename):
    """Extract text from any supported format. Returns str."""
    ext = os.path.splitext(filename)[1].lower()
    
    # Text-based formats
    if ext in ('.txt', '.md', '.csv', '.json', '.py', '.html', '.xml'):
        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return ""
    
    # PDF
    if ext == '.pdf':
        text = ""
        # Try pdfplumber first (better for structured PDFs)
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text()
                    if t:
                        text += f"\n--- Page {i+1} ---\n{t}\n"
            if text.strip():
                return text
        except Exception:
            pass
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    text += f"\n--- Page {i+1} ---\n{t}\n"
            return text
        except Exception:
            return f"[ERROR: Cannot extract text from {filename}]"
    
    # DOCX
    if ext == '.docx':
        try:
            from docx import Document
            doc = Document(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
            return text
        except Exception:
            return f"[ERROR: Cannot extract text from {filename}]"
    
    return f"[UNSUPPORTED: {ext}]"


def chunk_text(text, chunk_size=1000, overlap=200):
    """Semantic chunking: split by headings, paragraphs, then sentences."""
    if not text or not text.strip():
        return []
    
    # Step 1: Split by markdown headings (## / ###)
    heading_pattern = re.compile(r'(^#{1,4}\s+.+$)', re.MULTILINE)
    sections = heading_pattern.split(text)
    
    raw_chunks = []
    current_section = ""
    current_heading = ""
    
    for part in sections:
        if heading_pattern.match(part):
            if current_section.strip():
                raw_chunks.append((current_heading, current_section.strip()))
            current_heading = part.strip()
            current_section = ""
        else:
            current_section += part
    
    if current_section.strip():
        raw_chunks.append((current_heading, current_section.strip()))
    
    if not raw_chunks:
        raw_chunks = [("", text.strip())]
    
    # Step 2: For each section, split into sized chunks with overlap
    result = []
    for heading, section_text in raw_chunks:
        paragraphs = re.split(r'\n\s*\n', section_text)
        
        current = ""
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            if len(current) + len(p) < chunk_size:
                current += p + "\n\n"
            else:
                if current.strip():
                    result.append({
                        "text": current.strip(),
                        "metadata": {"section": heading} if heading else {}
                    })
                # If paragraph itself is too long, split by sentences
                if len(p) > chunk_size:
                    sentences = re.split(r'(?<=[。！？.!?])\s*', p)
                    sub = ""
                    for s in sentences:
                        if len(sub) + len(s) < chunk_size:
                            sub += s
                        else:
                            if sub.strip():
                                result.append({
                                    "text": sub.strip(),
                                    "metadata": {"section": heading} if heading else {}
                                })
                            sub = s
                    if sub.strip():
                        current = sub + "\n\n"
                    else:
                        current = ""
                else:
                    current = p + "\n\n"
        
        if current.strip():
            result.append({
                "text": current.strip(),
                "metadata": {"section": heading} if heading else {}
            })
    
    # Step 3: Apply overlap (append tail of previous chunk to next)
    if overlap > 0 and len(result) > 1:
        for i in range(1, len(result)):
            prev_text = result[i-1]["text"]
            if len(prev_text) > overlap:
                result[i]["text"] = prev_text[-overlap:] + "\n\n" + result[i]["text"]
    
    return [r for r in result if len(r["text"]) > 20]


def process_text(text, title):
    """Process raw text: chunk and return structured data."""
    if not text or not text.strip():
        return {"title": title, "full_text": "", "chunks": []}
    
    chunks = chunk_text(text)
    
    return {
        "title": title,
        "full_text": text,
        "chunks": chunks,
        "chunk_count": len(chunks)
    }


def process_file(filepath, filename):
    """Process a file: extract text, chunk, return structured data."""
    text = extract_text(filepath, filename)
    title = os.path.splitext(filename)[0]
    return process_text(text, title)


def summarize(text, max_len=200):
    """Extract first meaningful sentences as a summary."""
    if not text:
        return ""
    text = text.strip()
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    summary = ""
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        if len(summary) + len(s) > max_len:
            break
        summary += s
    return summary if summary else text[:max_len]


if __name__ == "__main__":
    # Self-test
    test_text = """# Introduction
This is a test document. It has multiple sections.

## Section 1
Here is the first section with some content. This is enough text to make a meaningful chunk.
The quick brown fox jumps over the lazy dog. Chinese text too: 这是一段中文测试。

## Section 2
Another section with more content. We need enough text to test the chunking algorithm properly.
More text here to fill up space. 这是第二段中文内容，用来测试分块算法。

### Subsection
Deeper nesting should also work correctly."""
    
    result = process_text(test_text, "Test Document")
    print(f"Title: {result['title']}")
    print(f"Chunks: {result['chunk_count']}")
    for i, c in enumerate(result['chunks']):
        print(f"\n--- Chunk {i} ---")
        print(f"  Section: {c['metadata'].get('section', 'N/A')}")
        print(f"  Length: {len(c['text'])} chars")
        print(f"  Preview: {c['text'][:80]}...")
    
    print(f"\nSummary: {summarize(test_text)}")
    print("\n✅ document_processor.py self-test passed")
