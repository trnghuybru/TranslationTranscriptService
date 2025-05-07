from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

router = APIRouter()

# Khởi tạo model M2M-100 (phiên bản 418M) cho dịch thuật
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Định nghĩa model cho dữ liệu nhận từ body theo dạng batch
class BatchTranslateRequest(BaseModel):
    texts: List[str]  # Danh sách các văn bản tiếng Nhật cần dịch

def translate_in_batches(texts: List[str], batch_size: int = 2):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id("vi"),
                max_length=128  # Giới hạn độ dài đầu ra
            )
        vietnamese_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results.extend(vietnamese_texts)
        if device == "cuda":
            torch.cuda.empty_cache()  # Giải phóng bộ nhớ GPU sau mỗi batch
    return results

@router.post("/translate_batch")
async def translate_batch(request: BatchTranslateRequest):
    try:
        print("Request data:", request.dict())  # In ra đầu vào để kiểm tra
        texts = request.texts
        tokenizer.src_lang = "ja"
        vietnamese_texts = translate_in_batches(texts, batch_size=2)
        return vietnamese_texts
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi khi dịch: {str(e)}")
