#!/usr/bin/env python3
"""Multimodal processor: video/audio transcription via Whisper, URL extraction via trafilatura."""
import os, subprocess, tempfile, random, requests


# ─── Whisper API Config ───
WHISPER_KEY = "app-ULzJbc3OaIN50mZVSU7sAa97"
WHISPER_BASE = "https://api.minimax.io"
WHISPER_URL = f"{WHISPER_BASE}/v1/audio/transcriptions"


def _random_suffix(ext):
    """Generate a random suffix for temp file naming."""
    return f"{random.randint(100000, 999999)}{ext}"


def video_to_text(filepath):
    """Extract audio from video and transcribe via Whisper API. Returns str."""
    try:
        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, _random_suffix(".wav"))

        # Convert video to WAV audio using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", filepath,
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return f"[ERROR: 视频转写失败: ffmpeg error: {result.stderr.strip()}]"

        if not os.path.exists(audio_path):
            return f"[ERROR: 视频转写失败: 音频文件未生成]"

        text = _whisper_audio(audio_path, "video")

        # Cleanup temp audio
        try:
            os.remove(audio_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

        return text

    except subprocess.TimeoutExpired:
        return "[ERROR: 视频转写失败: ffmpeg 超时]"
    except Exception as e:
        return f"[ERROR: 视频转写失败: {e}]"


def audio_to_text(filepath):
    """Transcribe audio file via Whisper API. Returns str."""
    try:
        return _whisper_audio(filepath, "audio")
    except Exception as e:
        return f"[ERROR: 音频转写失败: {e}]"


def _whisper_audio(filepath, source_type):
    """Send audio file to Whisper API. Returns transcribed text."""
    try:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f, "audio/wav")}
            data = {"model": "whisper-1"}
            resp = requests.post(
                WHISPER_URL,
                headers={"Authorization": f"Bearer {WHISPER_KEY}"},
                files=files,
                data=data,
                timeout=120
            )
        if resp.status_code != 200:
            try:
                err = resp.json()
                detail = err.get("error", {}).get("message", "") or err.get("message", "")
            except Exception:
                detail = resp.text[:200]
            return f"[ERROR: Whisper API {resp.status_code}: {detail}]"

        result = resp.json()
        text = result.get("text", "").strip()
        if not text:
            return f"[ERROR: {source_type} 转写返回空文本]"
        return text

    except requests.exceptions.Timeout:
        return f"[ERROR: Whisper API 超时]"
    except Exception as e:
        return f"[ERROR: Whisper 请求失败: {e}]"


def url_to_text(url):
    """Fetch URL and extract main content. Returns "{title}\n\n{content}" or error str."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        raw_html = resp.text
        content = None

        # Try trafilatura first
        try:
            import trafilatura
            content = trafilatura.extract(raw_html)
        except (ImportError, Exception):
            pass

        title = ""
        if not content:
            # Fallback to BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_html, "html.parser")

                # Try to get title
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                else:
                    h1 = soup.find("h1")
                    if h1:
                        title = h1.get_text(strip=True)

                # Extract paragraphs
                paragraphs = soup.find_all("p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
            except ImportError:
                return "[ERROR: URL解析失败: beautifulsoup4 未安装，请运行 pip install beautifulsoup4]"
            except Exception as e:
                return f"[ERROR: URL解析失败: BeautifulSoup error: {e}]"

        if not content or not content.strip():
            return f"[ERROR: URL解析失败: 无法提取正文内容]"

        result = f"{title}\n\n{content}" if title else content
        return result

    except requests.exceptions.Timeout:
        return "[ERROR: URL解析失败: 请求超时]"
    except requests.exceptions.HTTPError as e:
        return f"[ERROR: URL解析失败: HTTP {e.response.status_code}]"
    except Exception as e:
        return f"[ERROR: URL解析失败: {e}]"


def extract_duration(filepath):
    """Get video/audio duration in seconds using ffprobe. Returns float."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    # Self-test: check functions loadable, no syntax errors
    print("✅ multimodal_processor.py self-test passed (no runtime checks without real files)")
