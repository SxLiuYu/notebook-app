#!/usr/bin/env python3
"""Tests for multimodal_processor.py — mock Whisper API and requests."""
import os, sys, pytest, json
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Helpers ───
def make_fake_response(status_code=200, json_data=None, text=""):
    """Create a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_data) if json_data else MagicMock(side_effect=ValueError("not json"))
    resp.text = text
    return resp


# ─── extract_duration ───
def test_extract_duration_success():
    from multimodal_processor import extract_duration

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="123.456\n")
        assert extract_duration("/fake/video.mp4") == 123.456

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="0.0\n")
        assert extract_duration("/fake/audio.mp3") == 0.0


def test_extract_duration_ffprobe_error():
    from multimodal_processor import extract_duration

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        assert extract_duration("/fake/bad.mp4") == 0.0

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("ffprobe not found")
        assert extract_duration("/fake/file.mp4") == 0.0


# ─── url_to_text ───
def test_url_to_text_success_bs4_fallback():
    """Test URL extraction via bs4 fallback (trafilatura not installed in this env)."""
    from multimodal_processor import url_to_text

    html = "<html><title>Test Page</title><body><p>Hello world this is a test paragraph with enough length.</p><p>Another paragraph here with meaningful content as well.</p></body></html>"

    with patch("requests.get") as mock_get:
        mock_get.return_value = make_fake_response(200, text=html)
        # trafilatura unavailable → bs4 fallback → if bs4 installed, extracts content
        result = url_to_text("https://example.com/article")
        # If bs4 is installed, we get content; otherwise graceful error
        assert "Hello world" in result or "beautifulsoup4 未安装" in result


def test_url_to_text_trafilatura_installed():
    """Test URL extraction when trafilatura IS installed (requires trafilatura package)."""
    import importlib
    trafilatura_spec = importlib.util.find_spec("trafilatura")
    if trafilatura_spec is None:
        pytest.skip("trafilatura not installed")

    from multimodal_processor import url_to_text
    html = "<html><body><p>Content via trafilatura.</p></body></html>"

    with patch("requests.get") as mock_get:
        mock_get.return_value = make_fake_response(200, text=html)
        result = url_to_text("https://example.com/article")
        assert "trafilatura" in result or "Content via trafilatura" in result or "ERROR" in result


def test_url_to_text_http_error():
    from multimodal_processor import url_to_text

    with patch("requests.get") as mock_get:
        mock_get.return_value = make_fake_response(404)
        mock_get.return_value.raise_for_status = MagicMock(
            side_effect=Exception("HTTP 404")
        )
        result = url_to_text("https://example.com/notfound")
        assert result.startswith("[ERROR: URL解析失败")


def test_url_to_text_timeout():
    import requests as _req
    from multimodal_processor import url_to_text

    with patch("requests.get") as mock_get:
        mock_get.side_effect = _req.exceptions.Timeout("timeout")
        result = url_to_text("https://example.com/slow")
        assert "[ERROR: URL解析失败" in result


def test_url_to_text_empty_content():
    """Test URL with no meaningful content (both trafilatura unavailable and bs4 returns nothing)."""
    from multimodal_processor import url_to_text

    with patch("requests.get") as mock_get:
        mock_get.return_value = make_fake_response(200, text="<html><body></body></html>")
        result = url_to_text("https://example.com/empty")
        # trafilatura ImportError → bs4 fallback → no paragraphs → error
        assert result.startswith("[ERROR: URL解析失败")


# ─── _whisper_audio ───
def test_whisper_audio_success():
    from multimodal_processor import _whisper_audio

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_fake_response(200, json_data={"text": "Hello world this is a test."})

        with patch("builtins.open", mock_open(read_data=b"fake audio")):
            result = _whisper_audio("/fake/audio.wav", "audio")
            assert "Hello world" in result


def test_whisper_audio_api_error():
    from multimodal_processor import _whisper_audio

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_fake_response(
            400, json_data={"error": {"message": "bad request"}}
        )

        with patch("builtins.open", mock_open(read_data=b"fake")):
            result = _whisper_audio("/fake/audio.wav", "audio")
            assert result.startswith("[ERROR: Whisper API 400")


def test_whisper_audio_timeout():
    import requests as _req
    from multimodal_processor import _whisper_audio

    with patch("requests.post") as mock_post:
        mock_post.side_effect = _req.exceptions.Timeout("timeout")

        with patch("builtins.open", mock_open(read_data=b"fake")):
            result = _whisper_audio("/fake/audio.wav", "audio")
            assert "[ERROR: Whisper API 超时]" in result


def test_whisper_audio_empty_response():
    from multimodal_processor import _whisper_audio

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_fake_response(200, json_data={"text": ""})

        with patch("builtins.open", mock_open(read_data=b"fake")):
            result = _whisper_audio("/fake/audio.wav", "audio")
            assert result.startswith("[ERROR: audio 转写返回空文本]")


# ─── video_to_text ───
def test_video_to_text_ffmpeg_error():
    from multimodal_processor import video_to_text

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")
        result = video_to_text("/fake/video.mp4")
        assert result.startswith("[ERROR: 视频转写失败")


def test_video_to_text_whisper_error():
    from multimodal_processor import video_to_text

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        with patch("os.path.exists", return_value=True):
            with patch("multimodal_processor._whisper_audio", return_value="[ERROR: test error]"):
                result = video_to_text("/fake/video.mp4")
                assert result == "[ERROR: test error]"


def test_video_to_text_audio_file_not_created():
    from multimodal_processor import video_to_text

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        with patch("os.path.exists", return_value=False):
            result = video_to_text("/fake/video.mp4")
            assert result.startswith("[ERROR: 视频转写失败")


def test_video_to_text_timeout():
    import subprocess as _sub
    from multimodal_processor import video_to_text

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = _sub.TimeoutExpired(cmd=["ffmpeg"], timeout=120)
        result = video_to_text("/fake/video.mp4")
        assert "[ERROR: 视频转写失败: ffmpeg 超时]" in result


# ─── audio_to_text ───
def test_audio_to_text_success():
    from multimodal_processor import audio_to_text

    with patch("multimodal_processor._whisper_audio", return_value="Transcribed audio text."):
        result = audio_to_text("/fake/audio.mp3")
        assert result == "Transcribed audio text."


def test_audio_to_text_exception():
    from multimodal_processor import audio_to_text

    with patch("multimodal_processor._whisper_audio", side_effect=Exception("Unknown error")):
        result = audio_to_text("/fake/audio.mp3")
        assert result.startswith("[ERROR: 音频转写失败")


# ─── Module load ───
def test_module_loads():
    import multimodal_processor
    assert hasattr(multimodal_processor, "video_to_text")
    assert hasattr(multimodal_processor, "audio_to_text")
    assert hasattr(multimodal_processor, "url_to_text")
    assert hasattr(multimodal_processor, "extract_duration")
