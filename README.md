# Anuvaad-AI
This repository provides an advanced AI-powered pipeline for translating YouTube videos into multiple Indian and international languages. It uses Whisper for transcription, Google Gemini for translation and timing optimization, and Svara TTS for expressive, emotion-controlled speech synthesis. The workflow features chunk-based audio-video synchronization, parallel TTS generation, and automatic YouTube download, trim, and cleanup. The solution is designed for high-quality, natural-sounding translated videos with accurate timing and supports a wide range of languages.

# 🎬 YouTube Video Translation Pipeline (Chunk-Based, Svara TTS, Gemini)

This project provides an advanced AI-powered pipeline to translate YouTube videos into multiple Indian and international languages, using:
- **Whisper** for transcription
- **Google Gemini** for translation and timing optimization
- **Svara TTS** for expressive, emotion-controlled speech synthesis
- **Chunk-based workflow** for accurate audio-video sync

---

## 🚀 Features
- **Chunk-based audio sync**: Splits audio by pauses for natural speech timing
- **Parallel TTS**: Fast, multi-core audio generation for each chunk
- **Emotion & gender control**: Consistent voice and emotion across all chunks
- **Gemini-powered translation**: Optimizes translation for perfect video timing
- **Automatic YouTube download, trim, and cleanup**

---

## 🛠️ Setup
1. **Install Python 3.9+** and [FFmpeg](https://ffmpeg.org/download.html) (add to PATH)
2. **Create a virtual environment** (recommended):
	```powershell
	python -m venv .venv
	& .venv\Scripts\Activate.ps1
	```
3. **Install requirements:**
	```powershell
	pip install -r requirements.txt
	```
4. **(Optional) Set up GPU drivers** for PyTorch and CUDA for best performance

---

## ⚡ Usage

### 1. Edit `translate_video.py` to set:
	- `youtube_url` (YouTube video to translate)
	- `target_lang` (target language code, e.g. 'hi' for Hindi)
	- `GEMINI_API_KEY` (your Google Gemini API key)

### 2. Run the script
```powershell
python translate_video.py
```

### 3. Output
- The final translated video will be saved as `final_translated_video_chunked.mp4` in the project directory.

---

## 🧑‍💻 Main Pipeline (Chunk-Based)
1. Download and trim YouTube video (first 80 seconds)
2. Extract audio
3. Transcribe with Whisper (splits on pauses ≥ 1.5s)
4. Translate full text with Gemini (timing-optimized)
5. Map translation to chunks
6. Generate TTS audio for each chunk (parallel, emotion/gender controlled)
7. Assemble chunk audios and merge with video
8. Cleanup temporary files

---

## 🌍 Supported Languages

| Language   | Code | Language   | Code |
|------------|------|------------|------|
| Hindi      | hi   | Spanish    | es   |
| Marathi    | mr   | French     | fr   |
| Tamil      | ta   | German     | de   |
| Telugu     | te   | Italian    | it   |
| Bengali    | bn   | English    | en   |
| Gujarati   | gu   | Punjabi    | pa   |
| Kannada    | kn   | Urdu       | ur   |
| Malayalam  | ml   |            |      |

---

## 📝 Notes
- **Svara TTS** and **Whisper** models are loaded automatically (GPU/CPU auto-detect)
- **Gemini API key** is required for best translation quality and timing
- All temporary files are cleaned up automatically
- For best results, use a machine with a modern GPU and at least 16GB RAM

---

## 📄 Example

```python
# In translate_video.py:
youtube_url = "https://www.youtube.com/shorts/XcpTgxDEe6w"
target_lang = "hi"
GEMINI_API_KEY = "your-gemini-api-key"

final_video = youtube_video_translation_pipeline_chunked(
	 youtube_url,
	 target_lang,
	 GEMINI_API_KEY,
	 gender="Male",
	 emotion="Happy",
	 min_pause_duration=1.5,
	 parallel_tts=True
)
```

---

## 🧩 Advanced
- You can also use the classic pipeline (single audio, no chunking) or a simple pipeline (no Gemini) by calling the respective functions in `translate_video.py`.

---

## 🆘 Troubleshooting
- If Svara TTS fails to load, increase your Windows virtual memory (paging file) to 16GB+ and close other apps.
- If FFmpeg is not found, add it to your system PATH.
- For GPU acceleration, ensure CUDA drivers are installed and PyTorch is built with CUDA support.

---

## 📦 Requirements
See `requirements.txt` for all dependencies.
youtube_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Target language (use codes from table above)
target_lang = "hi"  # Hindi

# Optional: Add Gemini API Key for timing optimization
GEMINI_API_KEY = "your_api_key_here"
```

## 📦 Dependencies

All dependencies are installed in the virtual environment:
- torch (GPU version with CUDA 12.4)
- torchaudio
- transformers
- openai-whisper
- parler-tts
- moviepy
- yt-dlp
- deep-translator
- google-genai
- soundfile
- ffmpeg (system-level)

## 🔧 Troubleshooting

### FFmpeg not found
If you get "The system cannot find the file specified" error:
1. Close PowerShell completely
2. Open a new PowerShell window
3. Run the script again (FFmpeg PATH will be loaded)

Or use the provided `run_translation.ps1` script which handles this automatically.

### GPU not being used
Verify GPU is working:
```powershell
& 'E:/python/.venv/Scripts/python.exe' test_gpu.py
```

You should see:
```
✅ CUDA Available: True
✅ GPU Device: NVIDIA GeForce RTX 4060 Laptop GPU
✅ Whisper loaded on cuda
✅ Parler-TTS loaded on cuda:0
```

### Virtual environment issues
Reinstall dependencies:
```powershell
& 'E:/python/.venv/Scripts/python.exe' -m pip install -r requirements.txt
```

## 📊 Performance

With GPU acceleration:
- **Whisper transcription**: 5-10x faster than CPU
- **Parler-TTS generation**: 3-5x faster than CPU
- **1-minute video**: ~2-3 minutes total processing time

## 🎉 Output

The pipeline creates:
- `final_translated_video.mp4` - Your translated video with adjusted timing

Temporary files are automatically cleaned up after processing.

## 📝 Notes

- "Flash attention 2 is not installed" warning is normal and can be ignored
- Videos are automatically trimmed to first 60 seconds
- Video speed is adjusted to match translated audio duration
- Without Gemini API key, uses simple translation (may have timing mismatches)

---

**Created**: November 5, 2025
**Python**: 3.13.7
