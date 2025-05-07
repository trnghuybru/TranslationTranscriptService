from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mistralai import Mistral
import json
from settings import settings

router = APIRouter()

# Định nghĩa model Pydantic để validate dữ liệu đầu vào
class QuestionRequest(BaseModel):
    question: str

# API Key và Agent ID đặt trực tiếp trong code (chỉ dùng cho test)
MISTRAL_API_KEY = settings.MISTRAL_API_KEY
AGENT_ID = settings.AGENT_ID

# Khởi tạo Mistral Client
client = Mistral(api_key=MISTRAL_API_KEY)

@router.post("/ask")
async def ask_mistral(request: QuestionRequest):
    """
    Gửi câu hỏi đến Agent của Mistral AI và nhận phản hồi JSON đúng định dạng.
    """
    try:
        # Lấy câu hỏi từ request
        question = request.question
        
        # Gửi câu hỏi đến Mistral AI
        chat_response = client.agents.complete(
            agent_id=AGENT_ID,
            messages=[{"role": "user", "content": question}],
        )
        
        # Kiểm tra phản hồi hợp lệ
        if chat_response.choices:
            raw_response = chat_response.choices[0].message.content
            
            # Parse chuỗi JSON trả về từ Mistral thành object
            try:
                parsed_response = json.loads(raw_response)
                return parsed_response  # Trả về JSON đúng định dạng
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail="Phản hồi từ Mistral không phải JSON hợp lệ.")
        else:
            raise HTTPException(status_code=500, detail="Mistral AI không trả về kết quả hợp lệ.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi kết nối đến Mistral AI: {str(e)}")