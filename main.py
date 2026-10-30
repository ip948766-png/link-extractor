import os
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

# Plan tier size limits in MB
FREE_LIMIT_MB = int(os.getenv("FREE_LIMIT_MB", "25"))
PRO_LIMIT_MB = int(os.getenv("PRO_LIMIT_MB", "5120"))  # 5 GB

MAX_DURATION_SECONDS = int(os.getenv("MAX_DURATION_SECONDS", "10800"))
AUDIO_TTL_SECONDS = 1800

YOUTUBE_COOKIES_BASE64 = os.getenv("YOUTUBE_COOKIES_BASE64", "")
PROXY_URL = os.getenv("PROXY_URL", "")

CACHE_DIR = Path(tempfile.gettempdir()) / "transcriber-audio-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Audio Extraction API", version="3.0.0")

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
    plan: str = Field(default="free", description="User tier: 'free' (25MB limit) or 'pro' (5GB limit)")


class VideoDownloadRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)
    quality: str = Field(default="720", description="Preferred video quality: 360, 480, 720, 1080, or best")
    plan: str = Field(default="free", description="User tier: 'free' (25MB limit) or 'pro' (5GB limit)")


def validate_video_url(raw_url: str) -> str:
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
    """Remove old cached files and abandoned extraction directories."""
    now = time.time()

    try:
        for path in CACHE_DIR.iterdir():
            try:
                if now - path.stat().st_mtime > AUDIO_TTL_SECONDS:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not clean up: %s", path.name)
    except OSError:
def download_audio(url: str, output_dir: str, plan: str = "free") -> tuple[str, dict]:
    """Extract the audio track using yt-dlp and proxy/cookies if available."""

    max_mb = PRO_LIMIT_MB if plan.lower() == "pro" else FREE_LIMIT_MB
    max_bytes = max_mb * 1024 * 1024

    cookiefile_path = None

    if YOUTUBE_COOKIES_BASE64:
        try:
            cookie_data = base64.b64decode(
                YOUTUBE_COOKIES_BASE64,
                validate=True,
            ).decode("utf-8")

            cookiefile_path = os.path.join(output_dir, "cookies.txt")

            with open(cookiefile_path, "w", encoding="utf-8") as cookie_file:
                cookie_file.write(cookie_data)

        except Exception as exc:
            logger.warning("Could not load YouTube cookies: %s", exc)

    client_strategies = [
        ["tv_downgraded", "mweb", "android_vr"],
        ["android", "ios"],
        ["web_embedded", "mweb"],
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
            "max_filesize": max_bytes,
            "extractor_args": {
                "youtube": {
                    "player_client": client_list,
                }
            },
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        }

        if PROXY_URL:
            opts["proxy"] = PROXY_URL

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

                estimated_size = (
                    info.get("filesize")
                    or info.get("filesize_approx")
                )

                if estimated_size and estimated_size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio exceeds your {max_mb} MB tier limit.",
                    )

                ydl.download([url])

            candidates = [
                path
                for path in Path(output_dir).iterdir()
                if path.is_file()
                and path.suffix.lower() in AUDIO_EXTENSIONS
                and not path.name.endswith((".part", ".ytdl"))
                and path.name != "cookies.txt"
                and path.stat().st_size > 0
            ]

            if candidates:
                audio_file = max(
                    candidates,
                    key=lambda path: path.stat().st_size,
                )
                size = audio_file.stat().st_size

                if size <= max_bytes:
                    return str(audio_file), info

                raise HTTPException(
                    status_code=413,
                    detail=f"Audio exceeds your {max_mb} MB tier limit.",
                )

        except HTTPException:
            raise
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "yt-dlp strategy %s failed: %r",
                client_list,
                exc,
            )

    logger.error(
        "All audio extraction strategies failed: %r",
        last_exception,
    )

    raise HTTPException(
        status_code=502,
        detail=(
            "Could not extract audio from this video. "
            "For YouTube, check PROXY_URL or "
            "YOUTUBE_COOKIES_BASE64 in Render settings."
        ),
def download_video_file(url: str, output_dir: str, quality: str = "720", plan: str = "free") -> tuple[str, dict]:
    """Download a video file (MP4) with audio combined."""

    max_mb = PRO_LIMIT_MB if plan.lower() == "pro" else FREE_LIMIT_MB
    max_bytes = max_mb * 1024 * 1024

    cookiefile_path = None

    if YOUTUBE_COOKIES_BASE64:
        try:
            cookie_data = base64.b64decode(
                YOUTUBE_COOKIES_BASE64,
                validate=True,
            ).decode("utf-8")

            cookiefile_path = os.path.join(output_dir, "cookies.txt")

            with open(cookiefile_path, "w", encoding="utf-8") as cookie_file:
                cookie_file.write(cookie_data)

        except Exception as exc:
            logger.warning("Could not load YouTube cookies: %s", exc)

    if quality in ["360", "480", "720", "1080"]:
        format_str = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality}][ext=mp4]/best"
    else:
        format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    client_strategies = [
        ["tv_downgraded", "mweb", "android_vr"],
        ["android", "ios"],
        ["web_embedded", "mweb"],
    ]

    last_exception = None

    for client_list in client_strategies:
        opts = {
            "format": format_str,
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "retries": 3,
            "fragment_retries": 3,
            "max_filesize": max_bytes,
            "merge_output_format": "mp4",
            "extractor_args": {
                "youtube": {
                    "player_client": client_list,
                }
            },
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        }

        if PROXY_URL:
            opts["proxy"] = PROXY_URL

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

                estimated_size = (
                    info.get("filesize")
                    or info.get("filesize_approx")
                )

                if estimated_size and estimated_size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds your {max_mb} MB tier limit.",
                    )

                ydl.download([url])

            candidates = [
                path
                for path in Path(output_dir).iterdir()
                if path.is_file()
                and path.suffix.lower() in [".mp4", ".mkv", ".webm"]
                and not path.name.endswith((".part", ".ytdl"))
                and path.name != "cookies.txt"
                and path.stat().st_size > 0
            ]

            if candidates:
                video_file = max(
                    candidates,
                    key=lambda path: path.stat().st_size,
                )
                if video_file.stat().st_size <= max_bytes:
                    return str(video_file), info

                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds your {max_mb} MB tier limit.",
                )

        except HTTPException:
            raise
        except Exception as exc:
            last_exception = exc
            logger.warning("yt-dlp video strategy %s failed: %r", client_list, exc)

    raise HTTPException(
        status_code=502,
        detail="Could not download video. Check PROXY_URL or YOUTUBE_COOKIES_BASE64 in Render settings.",
    )


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "Audio Extraction API",
        "version": "3.2.0",
        "free_limit_mb": FREE_LIMIT_MB,
        "pro_limit_mb": PRO_LIMIT_MB,
    }


@app.post("/extract-audio")
async def handle_audio_extraction(request: LinkRequest):
    """Extract audio and return the binary file stream to the requesting backend."""

    cleanup_old_audio()
    validated_url = validate_video_url(request.url)

    session_dir = CACHE_DIR / os.urandom(16).hex()
    session_dir.mkdir(parents=True, exist_ok=False)

    try:
        audio_path, video_info = download_audio(
            validated_url,
            str(session_dir),
            plan=request.plan,
        )

        audio_file = Path(audio_path)
        mime_type, _ = mimetypes.guess_type(audio_file.name)

        if not mime_type or not mime_type.startswith("audio/"):
            mime_type = "application/octet-stream"

        duration = video_info.get("duration") or 0

        logger.info(
            "Audio extracted successfully: file=%s size=%s duration=%s",
            audio_file.name,
            audio_file.stat().st_size,
            duration,
        )

        return FileResponse(
            path=str(audio_file),
            media_type=mime_type,
            filename=audio_file.name,
            headers={
                "X-Video-Duration": str(duration),
                "X-Audio-Extraction-Status": "success",
            },
            background=BackgroundTask(
                shutil.rmtree,
                str(session_dir),
                ignore_errors=True,
            ),
        )

    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise


@app.post("/download-video")
async def handle_video_download(request: VideoDownloadRequest):
    """Download video and stream the MP4 file to the client."""

    cleanup_old_audio()
    validated_url = validate_video_url(request.url)

    session_dir = CACHE_DIR / os.urandom(16).hex()
    session_dir.mkdir(parents=True, exist_ok=False)

    try:
        video_path, video_info = download_video_file(
            validated_url,
            str(session_dir),
            quality=request.quality,
            plan=request.plan,
        )

        video_file = Path(video_path)
        mime_type, _ = mimetypes.guess_type(video_file.name)
        if not mime_type:
            mime_type = "video/mp4"

        duration = video_info.get("duration") or 0
        title = video_info.get("title") or "video"

        logger.info(
            "Returning video file: %s, size=%.2f MB, duration=%s seconds",
            video_file.name,
            video_file.stat().st_size / (1024 * 1024),
            duration,
        )

        return FileResponse(
            path=str(video_file),
            media_type=mime_type,
            filename=f"{title}.mp4",
            headers={
                "X-Video-Duration": str(duration),
                "X-Video-Title": title,
                "X-Video-Download-Status": "success",
            },
            background=BackgroundTask(
                shutil.rmtree,
                str(session_dir),
                ignore_errors=True,
            ),
        )

    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
