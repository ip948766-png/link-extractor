import os
import base64
import logging
import pathlib
import shutil
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
import yt_dlp

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcriber")

app = FastAPI(title="Media Downloader & Audio Extractor API", version="1.0.0")

# Directories
TEMP_DIR = pathlib.Path("/tmp/media_downloads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = pathlib.Path("/tmp/yt_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_FILE_PATH = CACHE_DIR / "youtube_cookies.txt"

# Configuration
YOUTUBE_COOKIES_BASE64 = os.getenv("YOUTUBE_COOKIES_BASE64", "").strip()
PROXY_URL = os.getenv("PROXY_URL", "").strip()

FREE_LIMIT_MB = 25
PRO_LIMIT_MB = 5000

# Cleanup interval (1 hour)
CLEANUP_THRESHOLD_SECONDS = 3600


def get_cookie_file_path() -> Optional[str]:
    """
    Initialize the YouTube cookie file from the current environment variable.
    Overwrites the file dynamically so env updates take immediate effect.
    """
    if not YOUTUBE_COOKIES_BASE64:
        logger.warning("YouTube cookies are not configured in environment variables.")
        return None

    try:
        cleaned_b64 = "".join(YOUTUBE_COOKIES_BASE64.split())
        cookie_bytes = base64.b64decode(cleaned_b64, validate=True)
        cookie_data = cookie_bytes.decode("utf-8")

        if not (
            "# Netscape HTTP Cookie File" in cookie_data
            or "# HTTP Cookie File" in cookie_data
        ):
            logger.warning("YouTube cookie file does not match Netscape format header.")
            return None

        if not any(
            domain in cookie_data
            for domain in [".youtube.com", "youtube.com", ".google.com"]
        ):
            logger.warning("Cookie file does not contain expected YouTube or Google domains.")
            return None

        COOKIE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIE_FILE_PATH.write_text(cookie_data, encoding="utf-8")

        logger.info(
            "YouTube cookies loaded successfully. File size: %s bytes",
            COOKIE_FILE_PATH.stat().st_size,
        )
        return str(COOKIE_FILE_PATH)
    except Exception as exc:
        logger.warning("Could not initialize YouTube cookies: %s", type(exc).__name__)
        return None


def cleanup_old_files():
    """Background task to remove files older than 1 hour."""
    now = time.time()
    for item in TEMP_DIR.glob("*"):
        if item.is_file() and (now - item.stat().st_mtime > CLEANUP_THRESHOLD_SECONDS):
            try:
                item.unlink()
                logger.info("Cleaned up old file: %s", item)
            except Exception as e:
                logger.error("Failed to delete %s: %s", item, e)
        elif item.is_dir() and (now - item.stat().st_mtime > CLEANUP_THRESHOLD_SECONDS):
            try:
                shutil.rmtree(item)
                logger.info("Cleaned up old directory: %s", item)
            except Exception as e:
                logger.error("Failed to delete directory %s: %s", item, e)


class ExtractAudioRequest(BaseModel):
    url: HttpUrl


class DownloadVideoRequest(BaseModel):
    url: HttpUrl
    quality: Optional[str] = "720"


def sanitize_header_value(value: str) -> str:
    """Sanitize string values so they can be safely encoded in HTTP headers (latin-1)."""
    if not value:
        return ""
    # Encode to latin-1 while replacing unencodable characters (like emojis), then decode back
    return value.encode("latin-1", errors="replace").decode("latin-1")


def download_audio(url: str, output_dir: str, plan: str = "free") -> tuple[str, dict]:
    """Extract audio using yt-dlp, enforcing plan tier size limits."""
    max_mb = PRO_LIMIT_MB if plan.lower() == "pro" else FREE_LIMIT_MB
    max_bytes = max_mb * 1024 * 1024

    cookiefile_path = get_cookie_file_path()

    logger.info("YouTube cookie file exists: %s", bool(cookiefile_path and os.path.isfile(cookiefile_path)))
    logger.info(
        "YouTube cookie file size: %s",
        os.path.getsize(cookiefile_path) if cookiefile_path and os.path.isfile(cookiefile_path) else 0,
    )
    logger.info("YouTube proxy configured: %s", bool(PROXY_URL))

    client_strategies = [
        ["mweb", "android"],
        ["ios", "web_embedded"],
        ["tv_embedded", "android_vr"],
    ]

    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    last_exception = None

    for clients in client_strategies:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "max_filesize": max_bytes,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "extractor_args": {
                "youtube": {
                    "player_client": clients,
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
            },
        }

        if cookiefile_path and os.path.exists(cookiefile_path):
            opts["cookiefile"] = cookiefile_path

        if PROXY_URL:
            opts["proxy"] = PROXY_URL

        try:
            with yt_dlp.YoutubeDL(opts) as ytdl:
                info = ytdl.extract_info(url, download=True)
                filename = ytdl.prepare_filename(info)
                # Convert extension to mp3 due to FFmpeg postprocessing
                base_name, _ = os.path.splitext(filename)
                mp3_filename = base_name + ".mp3"

                if os.path.exists(mp3_filename):
                    file_size = os.path.getsize(mp3_filename)
                    if file_size > max_bytes:
                        os.remove(mp3_filename)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Extracted audio size exceeds maximum allowed for {plan.upper()} tier ({max_mb}MB).",
                        )
                    metadata = {
                        "title": info.get("title", "Audio Track"),
                        "duration": info.get("duration"),
                        "uploader": info.get("uploader"),
                    }
                    return mp3_filename, metadata
        except HTTPException:
            raise
        except Exception as err:
            logger.warning("yt-dlp strategy %s failed: %r", clients, err)
            last_exception = err

    raise HTTPException(
        status_code=502,
        detail=f"Could not extract audio track from media link: {last_exception}",
    )


def download_video_file(url: str, output_dir: str, quality: str = "720", plan: str = "free") -> tuple[str, dict]:
    """Download a video file (MP4) with audio combined, enforcing tier size limits."""
    max_mb = PRO_LIMIT_MB if plan.lower() == "pro" else FREE_LIMIT_MB
    max_bytes = max_mb * 1024 * 1024

    cookiefile_path = get_cookie_file_path()

    logger.info("YouTube cookie file exists: %s", bool(cookiefile_path and os.path.isfile(cookiefile_path)))
    logger.info(
        "YouTube cookie file size: %s",
        os.path.getsize(cookiefile_path) if cookiefile_path and os.path.isfile(cookiefile_path) else 0,
    )
    logger.info("YouTube proxy configured: %s", bool(PROXY_URL))

    if quality in ["360", "480", "720", "1080"]:
        format_spec = f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best"
    else:
        format_spec = "bestvideo+bestaudio/best"

    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    client_strategies = [
        ["tv_embedded", "android_vr"],
        ["mweb", "android"],
        ["ios", "web_embedded"],
    ]

    last_exception = None

    for clients in client_strategies:
        opts = {
            "format": format_spec,
            "outtmpl": out_template,
            "max_filesize": max_bytes,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "extractor_args": {
                "youtube": {
                    "player_client": clients,
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
            },
        }

        if cookiefile_path and os.path.exists(cookiefile_path):
            opts["cookiefile"] = cookiefile_path

        if PROXY_URL:
            opts["proxy"] = PROXY_URL

        try:
            with yt_dlp.YoutubeDL(opts) as ytdl:
                info = ytdl.extract_info(url, download=True)
                filename = ytdl.prepare_filename(info)

                # Ensure filename ends with mp4 after merging
                base_name, _ = os.path.splitext(filename)
                mp4_filename = base_name + ".mp4"

                target_file = mp4_filename if os.path.exists(mp4_filename) else filename

                if os.path.exists(target_file):
                    file_size = os.path.getsize(target_file)
                    if file_size > max_bytes:
                        os.remove(target_file)
                        raise HTTPException(
                            status_code=413,
                            detail=f"Video size exceeds maximum allowed limit for {plan.upper()} tier ({max_mb}MB).",
                        )
                    metadata = {
                        "title": info.get("title", "Video Track"),
                        "duration": info.get("duration"),
                        "resolution": info.get("resolution"),
                        "uploader": info.get("uploader"),
                    }
                    return target_file, metadata
        except HTTPException:
            raise
        except Exception as err:
            logger.warning("yt-dlp video strategy %s failed: %r", clients, err)
            last_exception = err

    raise HTTPException(
        status_code=502,
        detail=f"Could not download video file from link: {last_exception}",
    )


@app.post("/extract-audio")
async def extract_audio_endpoint(
    req: ExtractAudioRequest,
    background_tasks: BackgroundTasks,
    x_user_plan: Optional[str] = Header(default="free"),
):
    """API endpoint to extract audio from video link."""
    background_tasks.add_task(cleanup_old_files)
    session_dir = TEMP_DIR / f"session_{int(time.time() * 1000)}"
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path, metadata = download_audio(str(req.url), str(session_dir), plan=x_user_plan)

    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename,
        headers={
            "X-Media-Title": sanitize_header_value(metadata.get("title", "")),
            "X-Media-Duration": str(metadata.get("duration", "")),
        },
    )


@app.post("/download-video")
async def download_video_endpoint(
    req: DownloadVideoRequest,
    background_tasks: BackgroundTasks,
    x_user_plan: Optional[str] = Header(default="free"),
):
    """API endpoint to download full video file."""
    background_tasks.add_task(cleanup_old_files)
    session_dir = TEMP_DIR / f"session_{int(time.time() * 1000)}"
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path, metadata = download_video_file(
        str(req.url), str(session_dir), quality=req.quality, plan=x_user_plan
    )

    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename,
        headers={
            "X-Media-Title": sanitize_header_value(metadata.get("title", "")),
            "X-Media-Duration": str(metadata.get("duration", "")),
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cookie_status = bool(COOKIE_FILE_PATH.exists() and COOKIE_FILE_PATH.stat().st_size > 0)
    return {
        "status": "healthy",
        "proxy_enabled": bool(PROXY_URL),
        "cookies_loaded": cookie_status,
    }
