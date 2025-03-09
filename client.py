import requests
import json
import time
import sys
import os

def transcribe_audio(api_url, file_path, options=None):
    """
    Transcribe an audio file using the Whisper API
    
    Args:
        api_url (str): Base URL of the API
        file_path (str): Path to the audio file
        options (dict): Transcription options
        
    Returns:
        dict: Transcription result
    """
    # Set default options if none provided
    if options is None:
        options = {
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "word_timestamps": True,
            "vad_filter": True
        }
    
    # Prepare the file upload
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        
        # Send the transcription request with options as form-data fields
        # Instead of sending options as a JSON string, send each option as a separate form field
        form_data = {}
        for key, value in options.items():
            form_data[key] = str(value).lower() if isinstance(value, bool) else str(value)
        
        print(f"Sending request to {api_url}/api/transcribe with options: {form_data}")
        response = requests.post(
            f"{api_url}/api/transcribe",
            files=files,
            data=form_data  # Send as individual form fields
        )
    
    if response.status_code != 202:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    # Get the task ID
    task = response.json()
    task_id = task["id"]
    print(f"Transcription task created: {task_id}")
    
    # Poll for the task status
    while True:
        response = requests.get(f"{api_url}/api/tasks/{task_id}")
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
        
        task = response.json()
        status = task["status"]
        
        if status == "completed":
            print("Transcription completed successfully!")
            return task["result"]
        elif status == "failed":
            print(f"Transcription failed: {task.get('error', 'Unknown error')}")
            return None
        
        print(f"Status: {status}. Waiting...")
        time.sleep(2)

def save_transcription(result, output_path):
    """
    Save transcription result to a file
    
    Args:
        result (dict): Transcription result
        output_path (str): Path to save the transcription
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Write full transcription
        full_text = " ".join([segment["text"] for segment in result["segments"]])
        f.write(f"# Full Transcription\n\n{full_text}\n\n")
        
        # Write segments with timestamps
        f.write("# Segments\n\n")
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
            
        # Write metadata
        f.write("\n# Metadata\n\n")
        f.write(f"Language: {result['language']} (probability: {result['language_probability']:.2f})\n")
        f.write(f"Processing time: {result['processing_time']:.2f} seconds\n")
        f.write(f"Audio duration: {result['audio_duration']:.2f} seconds\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client.py <api_url> <audio_file> [output_file]")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')  # Remove trailing slash if present
    audio_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "transcription.txt"
    
    # Transcribe the audio
    result = transcribe_audio(api_url, audio_file)
    
    if result:
        # Save the transcription
        save_transcription(result, output_file)
        print(f"Transcription saved to {output_file}")