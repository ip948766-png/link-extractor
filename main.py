from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp

app = FastAPI()

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
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        # ====================================================
        # FIX: CORRECT KEY FOR YT-DLP TO MASQUERADE AS CHROME
        # ====================================================
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,video/webm,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)
            raw_audio_url = info.get("url")
            
            if not raw_audio_url:
                raise HTTPException(status_code=400, detail="Could not extract streaming audio URL.")
                
            return {"status": "success", "audio_url": raw_audio_url}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
