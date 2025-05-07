from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import whisper
import yt_dlp
import os
import motor.motor_asyncio
from bson.objectid import ObjectId
from settings import settings

router = APIRouter()

# Kết nối MongoDB
MONGO_URL = "mongodb://admin:password@localhost:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.nihongocast
collection = db.subtitle

# Tải model Whisper
model = whisper.load_model("small")

# Model dữ liệu cho yêu cầu phiên âm
class TranscribeRequest(BaseModel):
    youtube_url: str

def extract_video_id(youtube_url: str) -> str:
    """Trích xuất video ID từ URL"""
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    return youtube_url.split("/")[-1]

async def download_audio(youtube_url: str) -> str:
    """Tải audio từ YouTube"""
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "downloaded_audio.%(ext)s",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return "downloaded_audio.mp3"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải audio: {str(e)}")

@router.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    youtube_url = request.youtube_url
    video_id = extract_video_id(youtube_url)

    # Kiểm tra xem transcript đã có trong MongoDB chưa
    existing_transcription = await collection.find_one({"video_id": video_id})
    if existing_transcription:
        return {"subtitles": existing_transcription["subtitles"]}

    # Nếu chưa có, tải audio và thực hiện transcript
    audio_path = await download_audio(youtube_url)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Không tìm thấy file âm thanh.")

    try:
        result = model.transcribe(audio_path, language="ja", word_timestamps=True)
        subtitles = [
            {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"]}
            for seg in result["segments"]
        ]

        # Lưu transcript vào MongoDB
        doc = {"video_id": video_id, "subtitles": subtitles}
        await collection.insert_one(doc)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phiên âm: {str(e)}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    return {"subtitles": subtitles}
