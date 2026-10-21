import os
import re
import uuid
import shutil
import logging
import tempfile
import ipaddress
import mimetypes
import time
import base64
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "24"))
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024
MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "10800"))
AUDIO_TTL_SECONDS = 1800

# Base64 encoded Netscape format YouTube cookie string (optional)
YOUTUBE_COOKIES_BASE64 = os.getenv("YOUTUBE_COOKIES_BASE64", "")

CACHE_DIR = Path(tempfile.gettempdir()) / "transcriber-audio-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Audio Extraction API", version="2.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

ALLOWED_DOMAINS = {
    "youtube.com",
    "youtube-nocookie.com",
    "youtu.be",
    "tiktok.com",
    "instagram.com",
    "instagr.am",
    "facebook.com",
    "fb.watch",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga",
    ".ogg", ".wav", ".webm", ".flac",
}


class LinkRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)


def validate_video_url(raw_url: str) -> str:
    """Validate and sanitize user provided video link."""
    cleaned_url = raw_url.strip()

    try:
        parsed = urlparse(cleaned_url)
        hostname = (parsed.hostname or "").lower().rstrip(".")

        if parsed.scheme != "https":
            raise ValueError("Only HTTPS links are accepted.")

        if not hostname or parsed.username or parsed.password:
            raise ValueError("Invalid URL.")

        if parsed.port not in (None, 443):
            raise ValueError("Unsupported URL port.")

        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise ValueError("IP-address URLs are not accepted.")

        allowed = any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in ALLOWED_DOMAINS
        )

        if not allowed:
            raise ValueError(
                "Supported platforms are YouTube, TikTok, "
                "Instagram, and Facebook."
            )

        return cleaned_url

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video URL: {exc}",
        )


def cleanup_old_audio():
    """Remove cache directories/files older than AUDIO_TTL_SECONDS."""
    now = time.time()

    try:
        for path in CACHE_DIR.iterdir():
            try:
                if (
                    path.is_file()
                    and now - path.stat().st_mtime > AUDIO_TTL_SECONDS
                ):
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not clean up cached file: %s", path.name
                )
    except OSError:
        logger.exception("Audio cache cleanup failed.")


def download_audio(url: str, output_dir: str) -> tuple[str, dict]:
    """Download audio track with bot detection bypass strategies for cloud servers."""
    
    # Handle optional youtube cookies passed via environment variable
    cookiefile_path = None
    if YOUTUBE_COOKIES_BASE64:
        try:
            cookie_data = base64.b64decode(YOUTUBE_COOKIES_BASE64).decode("utf-8")
            cookiefile_path = os.path.join(output_dir, "cookies.txt")
            with open(cookiefile_path, "w", encoding="utf-8") as f:
                f.write(cookie_data)
        except Exception as exc:
            logger.warning("Failed to parse YOUTUBE_COOKIES_BASE64: %s", exc)

    # Sequence of player clients to attempt when Cloud Datacenter IPs get flagged
    client_strategies = [
        ["tv_downgraded", "mweb", "android_vr"],
        ["android", "ios"],
        ["web_embedded", "mweb"]
    ]

    last_exception = None

    for client_list in client_strategies:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "max_filesize": MAX_AUDIO_BYTES,
            "extractor_args": {
                "youtube": {
                    "player_client": client_list
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
            "ignoreerrors": True
        }

        if cookiefile_path and os.path.exists(cookiefile_path):
            opts["cookiefile"] = cookiefile_path

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if not info:
                    continue

                duration = info.get("duration")
                if duration and duration > MAX_DURATION_SECONDS:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the configured duration limit.",
                    )

                estimated_size = info.get("filesize") or info.get("filesize_approx")
                if estimated_size and estimated_size > MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio exceeds the {MAX_AUDIO_MB} MB limit.",
                    )

                ydl.download([url])

            candidates = [
                path for path in Path(output_dir).iterdir()
                if path.is_file()
                and path.suffix.lower() in AUDIO_EXTENSIONS
                and not path.name.endswith((".part", ".ytdl"))
                and path.name != "cookies.txt"
            ]

            if candidates:
                audio_file = max(candidates, key=lambda path: path.stat().st_size)
                audio_path = str(audio_file)
                size = os.path.getsize(audio_path)

                if size > 0 and size <= MAX_AUDIO_BYTES:
                    return audio_path, info

        except HTTPException:
            raise
        except Exception as exc:
            last_exception = exc
            logger.warning("yt-dlp strategy %s failed: %r", client_list, exc)
            continue

    logger.exception("All yt-dlp extraction strategies failed: %r", last_exception)
    raise HTTPException(
        status_code=502,
        detail=(
            "YouTube bot verification check failed on host server. "
            "Please try again or configure YOUTUBE_COOKIES_BASE64 in your Render settings."
        )
    )


def transcribe_with_groq(audio_path: str) -> dict:
    """Send local audio file stream to Groq Audio Transcriptions API."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the Render server.",
        )

    size = os.path.getsize(audio_path)
    if size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {MAX_AUDIO_MB} MB limit for Groq upload.",
        )

    mime_type, _ = mimetypes.guess_type(audio_path)
    if not mime_type:
        mime_type = "audio/mpeg"

    try:
        filename = os.path.basename(audio_path)

        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (filename, audio_file, mime_type)
            }
            data = {
                "model": GROQ_MODEL,
                "response_format": "json"
            }
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}"
            }

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files
                )

                if response.status_code != 200:
                    logger.error("Groq API returned error: %s", response.text)
                    raise HTTPException(
                        status_code=502,
                        detail=f"Groq transcription failed: {response.text}"
                    )

                return response.json()

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error processing Groq transcription")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Groq pipeline error: {str(e)}"
        )


@app.post("/extract-audio")
async def handle_extraction_and_transcription(request: LinkRequest):
    """Main API endpoint to accept video URL, extract audio, and return transcript."""
    cleanup_old_audio()
    validated_url = validate_video_url(request.url)

    session_id = str(uuid.uuid4())
    runtime_dir = CACHE_DIR / session_id
    runtime_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio_path, video_info = download_audio(validated_url, str(runtime_dir))
        transcription_result = transcribe_with_groq(audio_path)

        return {
            "status": "success",
            "title": video_info.get("title", "Unknown Social Media Video"),
            "duration": video_info.get("duration", 0),
            "transcription": transcription_result.get("text", "")
        }
    finally:
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir, ignore_errors=True)
