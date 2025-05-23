from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List, Any
import os
import tempfile
import shutil
import json
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import base64
import io

load_dotenv()  # Thêm dòng này

# Tạo router thay vì Flask app
router = APIRouter()

# Biến toàn cục để lưu socket_manager
socket_manager = None

def setup_socket_handlers(socket_mgr):
    """Thiết lập socket_manager từ main.py"""
    global socket_manager
    socket_manager = socket_mgr

# Cấu hình các AI provider miễn phí
FREE_PROVIDERS = {
    "groq": {
        "name": "Groq (Miễn phí với Whisper + Llama)",
        "whisper_url": "https://api.groq.com/openai/v1/audio/transcriptions",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "whisper_model": "whisper-large-v3",
        "chat_model": "llama3-70b-8192"
    },
    "huggingface": {
        "name": "Hugging Face (Miễn phí)",
        "whisper_url": "https://api-inference.huggingface.co/models/openai/whisper-large-v3",
        "chat_url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",  # Model mới
        "api_key_env": "HUGGINGFACE_API_KEY",
        "whisper_model": "whisper-large-v3",
        "chat_model": "mistralai/Mistral-7B-Instruct-v0.3"  # Model chat phù hợp
    },
    "together": {
        "name": "Together AI ($5 credit miễn phí)",
        "whisper_url": "https://api.together.xyz/v1/audio/transcriptions", 
        "chat_url": "https://api.together.xyz/v1/chat/completions",
        "api_key_env": "TOGETHER_API_KEY",
        "whisper_model": "whisper-large-v3",
        "chat_model": "meta-llama/Llama-2-70b-chat-hf"
    },
    "deepseek": {
        "name": "DeepSeek (Rất rẻ, ~miễn phí)",
        "chat_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "chat_model": "deepseek-chat"
    }
}

PRIORITY = ["groq", "huggingface", "together", "deepseek"]  # Thứ tự ưu tiên mới

# Sửa phần chọn provider thành:
selected_provider = None
for provider_name in PRIORITY:
    config = FREE_PROVIDERS.get(provider_name)
    if not config:
        continue  # Bỏ qua nếu provider không tồn tại
    
    api_key = os.getenv(config["api_key_env"])
    if api_key:
        selected_provider = provider_name
        print(f"✅ Sử dụng {config['name']}")
        break

if not selected_provider:
    print("⚠️ Không tìm thấy API key nào, chạy trong chế độ LOCAL")

def convert_audio_to_wav(input_path, output_path):
    """Chuyển đổi audio sang format WAV"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Lỗi chuyển đổi audio: {e}")
        return False

def validate_audio_file(file_path):
    """Kiểm tra tính hợp lệ của file audio"""
    if not os.path.exists(file_path):
        return False, "File không tồn tại"
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File audio rỗng"
    
    if file_size < 1024:
        return False, "File audio quá nhỏ"
    
    try:
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000.0
        
        if duration < 0.5:
            return False, "Audio quá ngắn để phân tích"
        
        if duration > 600:
            return False, "Audio quá dài (tối đa 10 phút)"
            
        return True, f"Audio hợp lệ - Thời lượng: {duration:.2f}s"
        
    except Exception as e:
        return False, f"Không thể đọc file audio: {str(e)}"

def transcribe_with_groq(audio_path, api_key):
    """Chuyển đổi audio sang text bằng Groq Whisper"""
    try:
        url = FREE_PROVIDERS["groq"]["whisper_url"]
        headers = {"Authorization": f"Bearer {api_key}"}
        with open(audio_path, "rb") as f:
            files = {"file": f}
            data = {
                "model": FREE_PROVIDERS["groq"]["whisper_model"],
                "language": "ja"
            }
            
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "")
        else:
            print(f"Groq Whisper Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Lỗi Groq transcription: {e}")
        return None

def transcribe_with_huggingface(audio_path, api_key):
    """Chuyển đổi audio sang text bằng Hugging Face Whisper (MIỄN PHÍ)"""
    try:
        url = FREE_PROVIDERS["huggingface"]["whisper_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "audio/webm"  # Thêm Content-Type phù hợp
        }
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        response = requests.post(url, headers=headers, data=audio_bytes)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            elif isinstance(result, list) and len(result) > 0:
                return result[0].get("text", "")
        else:
            print(f"HuggingFace Whisper Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Lỗi HuggingFace transcription: {e}")
        return None

def analyze_with_groq(transcript, api_key):
    """Phân tích bằng Groq Llama (MIỄN PHÍ)"""
    try:
        url = FREE_PROVIDERS["groq"]["chat_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": FREE_PROVIDERS["groq"]["chat_model"],
            "messages": [
                {
                    "role": "system",
                    "content": """Bạn là trợ lý giáo viên tiếng Nhật. Hãy phân tích cuộc hội thoại theo các tiêu chí:

🌸 TỔNG QUAN:
- Chủ đề hội thoại: [Xác định chủ đề]
- Mức độ phù hợp: [Đánh giá theo trình độ học viên]

📖 PHÂN TÍCH CHI TIẾT:
1. Từ vựng:
   - Từ mới xuất hiện: [Liệt kê]
   - Lỗi dùng từ: [Ghi rõ từ sai và gợi ý từ đúng]

2. Ngữ pháp:
   - Cấu trúc đã sử dụng: [Liệt kê]
   - Lỗi ngữ pháp: [Chỉ ra lỗi và cách sửa]

3. Phát âm:
   - Từ phát âm khó: [Danh sách từ cần luyện tập]
   - Gợi ý luyện phát âm: [Phương pháp cụ thể]

4. Giao tiếp:
   - Độ trôi chảy: /10
   - Tự nhiên trong hội thoại: /10

💡 GỢI Ý CẢI THIỆN:
- Bài học nên ôn tập: [Liệt kê]
- Tài liệu tham khảo: [Sách/Website phù hợp]"""
                },
                {
                    "role": "user",
                    "content": f"Phân tích transcript cuộc họp:\n\n{transcript}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Groq Chat Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Lỗi Groq analysis: {e}")
        return None


def analyze_with_huggingface(transcript, api_key):
    try:
        url = FREE_PROVIDERS["huggingface"]["chat_url"]
        headers = {"Authorization": f"Bearer {api_key}"}
        
        payload = {
            "inputs": f"""<s>[INST] Bạn là trợ lý giáo viên tiếng Nhật. Hãy phân tích transcript sau:
            {transcript}
            Theo các tiêu chí: Từ vựng, Ngữ pháp, Phát âm. Phản hồi bằng tiếng Việt. [/INST]"""
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            print(f"HuggingFace Chat Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Lỗi HuggingFace analysis: {e}")
        return None

def analyze_with_deepseek(transcript, api_key):
    """Phân tích bằng DeepSeek (Rất rẻ)"""
    try:
        url = FREE_PROVIDERS["deepseek"]["chat_url"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": FREE_PROVIDERS["deepseek"]["chat_model"],
            "messages": [
                {
                    "role": "system",
                    "content": """Bạn là AI chuyên phân tích cuộc họp. Phản hồi bằng tiếng Việt với format:

📋 TÓM TẮT CUỘC HỌP:
[Tóm tắt chính của cuộc họp]

🎯 CÁC ĐIỂM CHÍNH:
[Liệt kê các điểm quan trọng]

📌 QUYẾT ĐỊNH/HÀNH ĐỘNG:
[Các quyết định được đưa ra]

👥 THÔNG TIN THÊM:
[Thông tin bổ sung nếu có]"""
                },
                {
                    "role": "user", 
                    "content": f"Hãy phân tích transcript cuộc họp sau:\n\n{transcript}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"DeepSeek Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Lỗi DeepSeek analysis: {e}")
        return None

@router.post('/analyze')
async def analyze_audio(audio: UploadFile = File(...)):
    temp_dir = None
    try:
        print(f"🎵 Nhận request phân tích audio - Provider: {selected_provider or 'LOCAL'}")
        
        if not audio:
            raise HTTPException(status_code=400, detail="Không tìm thấy file audio")
            
        print(f"📁 File: {audio.filename}, Type: {audio.content_type}")
        
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Tên file audio rỗng")
        
        # Tạo thư mục tạm và lưu file
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, f"original_{audio.filename}")
        
        # Lưu file từ FastAPI UploadFile
        with open(original_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
            # Đặt lại con trỏ file để có thể đọc lại nếu cần
            await audio.seek(0)
        
        print(f"💾 Đã lưu file: {os.path.getsize(original_path)} bytes")
        
        # Validate file
        is_valid, message = validate_audio_file(original_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"File audio không hợp lệ: {message}")
        
        print(f"✅ {message}")
        
        # Chuyển đổi sang WAV
        wav_path = os.path.join(temp_dir, "converted.wav")
        if not convert_audio_to_wav(original_path, wav_path):
            raise HTTPException(status_code=500, detail="Không thể chuyển đổi audio")
        
        print(f"🔄 Đã chuyển đổi sang WAV: {os.path.getsize(wav_path)} bytes")

        # Xử lý theo provider
        if not selected_provider:
            # LOCAL MODE
            audio_info = AudioSegment.from_wav(wav_path)
            duration = len(audio_info) / 1000.0
            
            summary = f"""📊 Phân tích cuộc họp (Chế độ Offline)

⏱️ Thông tin audio:
- Thời lượng: {duration:.2f} giây  
- Kích thước: {os.path.getsize(wav_path)} bytes
- Định dạng: WAV, 16kHz, Mono

💡 Kết quả:
Audio đã được ghi và xử lý thành công trong chế độ offline.

🔧 Để sử dụng AI miễn phí:
1. GROQ API (MIỄN PHÍ): Đăng ký tại https://console.groq.com
2. HuggingFace (MIỄN PHÍ): Tạo token tại https://huggingface.co/settings/tokens  
3. Together AI ($5 credit): https://api.together.xyz
4. DeepSeek (Rất rẻ): https://platform.deepseek.com

Thiết lập biến môi trường:
- GROQ_API_KEY=your_groq_key
- HUGGINGFACE_API_KEY=your_hf_token
- TOGETHER_API_KEY=your_together_key
- DEEPSEEK_API_KEY=your_deepseek_key

🎉 Hệ thống ghi âm hoạt động bình thường!"""
            
        else:
            # ONLINE MODE với Free APIs
            api_key = os.getenv(FREE_PROVIDERS[selected_provider]["api_key_env"])
            
            # Step 1: Transcribe audio
            print("🎙️ Đang chuyển đổi audio sang text...")
            transcript = None
            
            if selected_provider == "groq":
                transcript = transcribe_with_groq(wav_path, api_key)
            elif selected_provider == "huggingface":
                transcript = transcribe_with_huggingface(wav_path, api_key)
            # Together và DeepSeek có thể không có Whisper, fallback về Groq hoặc HF
            
            if not transcript:
                raise HTTPException(
                    status_code=500, 
                    detail="Không thể chuyển đổi audio sang text. Kiểm tra API key hoặc thử lại."
                )
            
            print(f"📝 Transcript: {len(transcript)} ký tự")
            print(f"📝 Nội dung: {transcript[:200]}...")
            
            if len(transcript.strip()) < 10:
                summary = f"""📊 Kết quả phân tích cuộc họp

⚠️ Cảnh báo: Transcript quá ngắn hoặc không rõ ràng

📝 Nội dung nhận diện: "{transcript}"

💡 Gợi ý cải thiện:
- Kiểm tra microphone hoạt động tốt
- Nói to và rõ ràng hơn  
- Ghi âm trong môi trường ít tiếng ồn
- Đảm bảo kết nối internet ổn định

🔧 Provider đang dùng: {FREE_PROVIDERS[selected_provider]['name']}"""
            else:
                # Step 2: Analyze với AI
                print("🧠 Đang phân tích nội dung...")
                analysis = None
                
                if selected_provider == "groq":
                    analysis = analyze_with_groq(transcript, api_key)
                elif selected_provider == "huggingface":
                    analysis = analyze_with_huggingface(transcript, api_key)
                elif selected_provider == "deepseek":
                    analysis = analyze_with_deepseek(transcript, api_key)
                # Có thể thêm các provider khác
                
                if analysis:
                    summary = f"""🎯 Sử dụng {FREE_PROVIDERS[selected_provider]['name']}

{analysis}

---
📝 Transcript gốc:
{transcript}

✅ Phân tích hoàn thành bằng AI miễn phí!"""
                else:
                    summary = f"""📊 Kết quả chuyển đổi audio

📝 Nội dung cuộc họp:
{transcript}

⚠️ Lưu ý: Chỉ thực hiện chuyển đổi audio sang text.
Phân tích AI gặp lỗi, vui lòng thử lại.

🔧 Provider: {FREE_PROVIDERS[selected_provider]['name']}"""

        print("✅ Trả về kết quả thành công")
        return {"summary": summary}

    except Exception as e:
        print(f"❌ Lỗi tổng quát: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("🗑️ Đã xóa thư mục tạm")
            except:
                pass

@router.get('/health')
async def health_check():
    """Endpoint kiểm tra trạng thái server"""
    available_providers = []
    for name, config in FREE_PROVIDERS.items():
        api_key = os.getenv(config["api_key_env"])
        if api_key:
            available_providers.append({
                "name": name,
                "display_name": config["name"],
                "status": "available"
            })
    
    status = {
        "status": "healthy",
        "mode": "online" if selected_provider else "local",
        "current_provider": selected_provider,
        "available_providers": available_providers,
        "setup_guide": {
            "groq": "Miễn phí tại https://console.groq.com",
            "huggingface": "Miễn phí tại https://huggingface.co/settings/tokens",
            "together": "$5 credit tại https://api.together.xyz", 
            "deepseek": "Rất rẻ tại https://platform.deepseek.com"
        }
    }
    return status

@router.get('/providers')
async def list_providers():
    """Liệt kê các provider AI miễn phí"""
    providers_info = []
    for name, config in FREE_PROVIDERS.items():
        api_key = os.getenv(config["api_key_env"])
        providers_info.append({
            "name": name,
            "display_name": config["name"],
            "status": "configured" if api_key else "not_configured",
            "env_var": config["api_key_env"],
            "features": {
                "whisper": "whisper_url" in config,
                "chat": "chat_url" in config
            }
        })
    
    return {
        "providers": providers_info,
        "current": selected_provider,
        "instructions": {
            "groq": {
                "url": "https://console.groq.com",
                "description": "Hoàn toàn miễn phí, rất nhanh, hỗ trợ Whisper + Llama",
                "setup": "1. Đăng ký tài khoản\n2. Tạo API key\n3. Set GROQ_API_KEY=your_key"
            },
            "huggingface": {
                "url": "https://huggingface.co/settings/tokens",
                "description": "Miễn phí nhưng có giới hạn tốc độ",
                "setup": "1. Tạo tài khoản HF\n2. Tạo Access Token\n3. Set HUGGINGFACE_API_KEY=your_token"
            },
            "together": {
                "url": "https://api.together.xyz",
                "description": "$5 credit miễn phí khi đăng ký",
                "setup": "1. Đăng ký và nhận $5\n2. Tạo API key\n3. Set TOGETHER_API_KEY=your_key"
            },
            "deepseek": {
                "url": "https://platform.deepseek.com",
                "description": "Rất rẻ, chất lượng cao (~$0.001/1000 tokens)",
                "setup": "1. Đăng ký tài khoản\n2. Nạp ít tiền ($1-5)\n3. Set DEEPSEEK_API_KEY=your_key"
            }
        }
    }

# Không cần phần if __name__ == "__main__" vì đây là router, không phải ứng dụng chính
