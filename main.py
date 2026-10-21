
import os
import re
import uuid
import shutil
import logging
import tempfile
import ipaddress
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yt_dlp

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl

# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")

# Conservative default for Groq's 25 MB file-upload tier.
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "24"))
MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024
MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "10800"))
AUDIO_TTL_SECONDS = 1800

CACHE_DIR = Path(tempfile.gettempdir()) / "transcriber-audio-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Audio Extraction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Only accept URLs from the supported platforms.
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


# --------------------------------------------------
# URL validation
# --------------------------------------------------

def validate_video_url(raw_url: str) -> str:
    try:
        parsed = urlparse(raw_url.strip())
        hostname = (parsed.hostname or "").lower().rstrip(".")

        if parsed.scheme != "https":
            raise ValueError("Only HTTPS links are accepted.")

        if not hostname or parsed.username or parsed.password:
            raise ValueError("Invalid URL.")

        if parsed.port not in (None, 443):
            raise ValueError("Unsupported URL port.")

        # Reject IP addresses and non-public address literals.
        try:
            ipaddress.ip_address(hostname)
            raise ValueError("IP-address URLs are not accepted.")
        except ValueError as exc:
            if "not accepted" in str(exc):
                raise

        allowed = any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in ALLOWED_DOMAINS
        )

        if not allowed:
            raise ValueError(
                "Supported platforms are YouTube, TikTok, "
                "Instagram, and Facebook."
            )

        return raw_url.strip()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video URL: {exc}",
        )


# --------------------------------------------------
# Temporary audio-file cleanup
# --------------------------------------------------

def cleanup_old_audio():
    try:
        import time

        now = time.time()

        for path in CACHE_DIR.iterdir():
            try:
                if now - path.stat().st_mtime > AUDIO_TTL_SECONDS:
                    if path.is_file():
                        path.unlink(missing_ok=True)
            except OSError:
                pass

    except OSError:
        logger.warning("Audio cache cleanup failed.")


# --------------------------------------------------
# Download audio using yt-dlp
# --------------------------------------------------

def download_audio(url: str, output_dir: str) -> tuple[str, dict]:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "retries": 1,
        "fragment_retries": 1,
        "max_filesize": MAX_AUDIO_BYTES,
        "impersonate": "chrome",
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                raise HTTPException(
                    status_code=422,
                    detail="The platform did not return video information.",
                )

            duration = info.get("duration")

            if duration and duration > MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail="Video exceeds the configured duration limit.",
                )

            estimated_size = (
                info.get("filesize")
                or info.get("filesize_approx")
            )

            if estimated_size and estimated_size > MAX_AUDIO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio exceeds the {MAX_AUDIO_MB} MB limit.",
                )

            # Download the actual audio instead of returning an
            # expiring third-party streaming URL.
            ydl.download([url])

        candidates = [
            p for p in Path(output_dir).iterdir()
            if p.is_file()
            and p.suffix.lower() in AUDIO_EXTENSIONS
            and not p.name.endswith((".part", ".ytdl"))
        ]

        if not candidates:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Audio extraction produced no supported audio file. "
                    "The video may be restricted or the platform may "
                    "have changed its extraction requirements."
                ),
            )

        audio_path = str(max(candidates, key=lambda p: p.stat().st_size))
        size = os.path.getsize(audio_path)

        if size == 0:
            raise HTTPException(
                status_code=422,
                detail="The extracted audio file is empty.",
            )

        if size > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds the {MAX_AUDIO_MB} MB limit.",
            )

        return audio_path, info

    except HTTPException:
        raise
    except Exception as exc:
        # Keep detailed diagnostics in Render logs, not public responses.
        logger.exception("yt-dlp extraction failed: %s", type(exc).__name__)

        message = str(exc).lower()

        if "private" in message or "login" in message or "sign in" in message:
            detail = (
                "This video may require login or may be private. "
                "Try a publicly accessible video."
            )
        elif "unsupported url" in message:
            detail = "This URL is not supported by the installed yt-dlp version."
        elif "403" in message or "406" in message or "forbidden" in message:
            detail = (
                "The platform rejected the extraction request. "
                "It may require authentication or have changed its "
                "access controls. Check the Render logs."
            )
        elif "too large" in message or "max-filesize" in message:
            detail = f"Audio exceeds the {MAX_AUDIO_MB} MB limit."
        else:
            detail = (
                "Audio extraction failed. The video may be unavailable, "
                "restricted, or temporarily blocked. Check Render logs."
            )

        raise HTTPException(status_code=502, detail=detail)


# --------------------------------------------------
# Groq Whisper transcription
# --------------------------------------------------

def transcribe_with_groq(audio_path: str) -> dict:
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the Render server.",
        )

    size = os.path.getsize(audio_path)

    if size > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {MAX_AUDIO_MB} MB limit.",
        )

    filename = Path(audio_path).name
    content_type = (
        mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )

    try:
        with open(audio_path, "rb") as audio_file:
            response = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                data={
                    "model": GROQ_MODEL,
                    "response_format": "json",
                    "temperature": "0",
                },
                files={
                    "file": (filename, audio_file, content_type),
                },
                timeout=300.0,
            )

        if response.is_error:
            logger.error(
                "Groq returned HTTP %s: %s",
                response.status_code,
                response.text[:1000],
            )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=502,
                    detail="Groq rejected the API key. Check GROQ_API_KEY.",
                )

            if response.status_code == 413:
                raise HTTPException(
                    status_code=413,
                    detail="Groq rejected the audio because it is too large.",
                )

            if response.status_code == 429:
                raise HTTPException(
                    status_code=503,
                    detail="Groq rate limit or quota reached. Try again later.",
                )

            raise HTTPException(
                status_code=502,
                detail=f"Groq transcription failed (HTTP {response.status_code}).",
            )

        result = response.json()

        return {
            "status": "success",
            "text": result.get("text", ""),
            "model": GROQ_MODEL,
        }

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Groq transcription timed out. Try again.",
        )
    except httpx.RequestError:
        logger.exception("Could not connect to Groq.")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the Groq transcription service.",
        )
    except Exception:
        logger.exception("Unexpected Groq response error.")
        raise HTTPException(
            status_code=502,
            detail="Unexpected error while processing the Groq response.",
        )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "audio-extraction",
        "platforms": [
            "YouTube",
            "TikTok",
            "Instagram",
            "Facebook",
        ],
    }


@app.post("/extract-audio")
def extract_audio(request: LinkRequest, http_request: Request):
    """
    Backwards-compatible extraction endpoint.

    Returns status + audio_url. The URL points to a temporary
    audio file hosted by this backend, not a third-party stream.
    """
    cleanup_old_audio()
    url = validate_video_url(request.url)

    with tempfile.TemporaryDirectory(prefix="extract-") as temp_dir:
        audio_path, info = download_audio(url, temp_dir)

        suffix = Path(audio_path).suffix.lower()
        token = uuid.uuid4().hex
        cached_path = CACHE_DIR / f"{token}{suffix}"

        shutil.copyfile(audio_path, cached_path)

    return {
        "status": "success",
        "audio_url": str(http_request.base_url).rstrip("/")
        + f"/audio/{token}",
        "title": info.get("title"),
        "duration": info.get("duration"),
        "expires_in": AUDIO_TTL_SECONDS,
    }


@app.get("/audio/{token}")
def serve_audio(token: str):
    # Only random tokens created by this service are accepted.
    if not re.fullmatch(r"[a-f0-9]{32}", token):
        raise HTTPException(status_code=404, detail="Audio not found.")

    matches = list(CACHE_DIR.glob(f"{token}.*"))

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Audio expired or not found. Extract the link again.",
        )

    path = matches[0]

    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Audio not found.")

    media_type = (
        mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
    )


@app.post("/transcribe-link")
def transcribe_link(request: LinkRequest):
    """
    Complete flow: validate URL -> download audio -> Groq Whisper.
    The audio file is removed automatically after this request.
    """
    cleanup_old_audio()
    url = validate_video_url(request.url)

    with tempfile.TemporaryDirectory(prefix="transcribe-") as temp_dir:
        audio_path, info = download_audio(url, temp_dir)
        result = transcribe_with_groq(audio_path)

        result["title"] = info.get("title")
        result["duration"] = info.get("duration")

        return result
