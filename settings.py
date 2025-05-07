import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Cấu hình cho ứng dụng"""
    MONGODB_URL = os.getenv("MONGODB_URL")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    AGENT_ID = os.getenv("AGENT_ID")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

settings = Settings()