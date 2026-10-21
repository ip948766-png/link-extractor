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
        # Natively force browser impersonation to clear out 406 errors
        'impersonate': 'chrome',
        'ignoreerrors': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=False)
            
            if not info:
                raise HTTPException(status_code=400, detail="Platform blocked connection. Try another link layout.")
                
            raw_audio_url = info.get("url")
            
            if not raw_audio_url:
                raise HTTPException(status_code=400, detail="Could not clean raw streaming link.")
                
            return {"status": "success", "audio_url": raw_audio_url}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
