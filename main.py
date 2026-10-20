from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp

app = FastAPI()

# Enable CORS so your main app can securely fetch data from this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LinkRequest(BaseModel):
    url: str

@app.post("/extract-audio")
def extract_audio(request: LinkRequest):
    ydl_opts = {
        'format': 'bestaudio/best',  # Targets only the raw audio stream to keep it fast
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # download=False fetches just the streaming media URL metadata without saving to disk
            info = ydl.extract_info(request.url, download=False) 
            raw_audio_url = info.get("url")

            if not raw_audio_url:
                raise HTTPException(status_code=400, detail="Could not extract streaming audio URL.")

            return {"status": "success", "audio_url": raw_audio_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
