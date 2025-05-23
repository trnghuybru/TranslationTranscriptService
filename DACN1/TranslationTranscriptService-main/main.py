from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transcription_service import router as transcription_router
from translation_service import router as translation_router
from chatbot_service import router as chatbot_router
from groupkaiwa import router as groupkaiwa_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bạn có thể giới hạn theo domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router của các service với prefix riêng để dễ quản lý
app.include_router(transcription_router, prefix="/api")
app.include_router(translation_router, prefix="/api")
app.include_router(chatbot_router, prefix="/api")
app.include_router(groupkaiwa_router, prefix="/api")



