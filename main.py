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
