# INSTALL REQUIRED PACKAGES
#!pip install moviepy whisper deep-translator elevenlabs google-genai yt-dlp snac ipywidgets torch transformers soundfile librosa

import subprocess
import os
import math
import google.generativeai as genai  # Fixed import
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import numpy as np
import librosa
import time
import concurrent.futures
from IPython.display import Audio, display


# YOUTUBE DOWNLOAD SECTION
import yt_dlp

def download_youtube_video(youtube_url, output_path="downloaded_video.mp4"):
    """Download YouTube video and return the path"""
    print("📥 Downloading YouTube video...")

    ydl_opts = {
        'format': 'best[height<=720]/best',  # Prefer up to 720p
        'outtmpl': output_path,
        'quiet': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print("✅ YouTube video downloaded successfully!")
        return output_path
    except Exception as e:
        print(f"❌ YouTube download failed: {e}")
        raise

def trim_video_to_first_minute(video_path, output_path="trimmed_video.mp4"):
    """Trim video to first 80 seconds"""
    print("✂ Trimming video to first 80 seconds...")

    # Use FFmpeg to trim the video
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', '0',  # Start from beginning
        '-t', '80',  # Duration of 80 seconds
        '-c', 'copy',  # Copy streams without re-encoding
        '-y',  # Overwrite output file
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Video trimmed successfully to first 80 seconds!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Video trimming failed: {e}")
        # Fallback using moviepy
        print("🔄 Using MoviePy fallback for trimming...")
        from moviepy.editor import VideoFileClip

        video = VideoFileClip(video_path)
        trimmed_video = video.subclip(0, min(80, video.duration))
        trimmed_video.write_videofile(output_path, verbose=False, logger=None, codec='libx264')
        video.close()
        trimmed_video.close()
        print("✅ Video trimmed successfully (MoviePy fallback)!")
        return output_path

# 1. VIDEO PROCESSING SECTION
from moviepy.editor import VideoFileClip, AudioFileClip

def get_video_duration(video_path):
    """Get the duration of the video in seconds"""
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()
    return duration

def extract_audio_from_video(video_path):
    """Extract audio from MP4 video file"""
    print("🎬 Extracting audio from video...")
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.mp3"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    print("✅ Audio extracted successfully!")
    return audio_path

def slow_down_audio_for_transcription(audio_path, language):
    """Slow down audio if language is not English to help transcription"""
    if language == 'en':
        print("✅ Audio is in English, no speed adjustment needed for transcription")
        return audio_path

    print(f"🔍 Non-English audio detected ({language}), slowing down for better transcription...")

    # Slow down audio by 0.8x (20% slower) for better transcription
    slowed_audio_path = "slowed_audio_for_transcription.mp3"

    cmd = [
        'ffmpeg',
        '-i', audio_path,
        '-filter:a', 'atempo=0.8',
        '-y',
        slowed_audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Audio slowed down successfully for transcription!")
        return slowed_audio_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Audio slowing failed: {e}")
        # Fallback - use original audio
        return audio_path

def adjust_video_speed(video_path, original_duration, target_duration, output_path):
    """Adjust video speed to match target duration"""
    print(f" Adjusting video speed: {original_duration:.2f}s → {target_duration:.2f}s")

    # Calculate speed multiplier (how much to speed up/slow down)
    # If audio is shorter than video: speed_multiplier > 1 (speed up)
    # If audio is longer than video: speed_multiplier < 1 (slow down)
    speed_multiplier = original_duration / target_duration

    # Limit speed adjustment to reasonable range (0.5x to 2x)
    speed_multiplier = max(0.5, min(2.0, speed_multiplier))

    if abs(speed_multiplier - 1.0) < 0.05:  # Less than 5% difference, no need to adjust
        print(" Speed adjustment negligible, using original video")
        # Just copy the file
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

    print(f"🎚 Applying video speed multiplier: {speed_multiplier:.2f}x")
    
    # For FFmpeg setpts filter: lower values = faster playback
    # setpts = 1/speed_multiplier
    # Example: speed_multiplier=2 → setpts=0.5 → 2x faster
    setpts_value = 1.0 / speed_multiplier

    # Use FFmpeg for high-quality speed adjustment
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-filter:v', f'setpts={setpts_value}*PTS',
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Video speed adjusted successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Video speed adjustment failed: {e}")
        # Fallback - just use original video
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

def merge_audio_with_video(adjusted_video_path, new_audio_path, output_path):
    """Merge new audio with adjusted video using FFmpeg directly to avoid warnings"""
    print(" Merging new audio with adjusted video...")

    # Use FFmpeg directly to avoid MoviePy warnings
    cmd = [
        'ffmpeg',
        '-i', adjusted_video_path,  # Input video (already speed-adjusted)
        '-i', new_audio_path,       # Input audio
        '-c:v', 'copy',             # Copy video stream without re-encoding
        '-c:a', 'aac',              # Encode audio to AAC
        '-map', '0:v:0',           # Use video from first input
        '-map', '1:a:0',           # Use audio from second input
        '-shortest',               # End when the shortest stream ends
        '-y',                      # Overwrite output
        output_path
    ]

    try:
        # Run FFmpeg and suppress warnings by capturing output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Video with translated audio created successfully!")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg merge failed: {e}")
        print(" Falling back to MoviePy...")

        # Fallback to MoviePy
        video = VideoFileClip(adjusted_video_path)
        new_audio = AudioFileClip(new_audio_path)

        # Set the new audio to match video duration
        if new_audio.duration > video.duration:
            new_audio = new_audio.subclip(0, video.duration)

        final_video = video.set_audio(new_audio)
        final_video.write_videofile(output_path, verbose=False, logger=None, codec='libx264')

        video.close()
        new_audio.close()
        final_video.close()

        print("✅ Video with translated audio created (MoviePy fallback)!")
        return output_path

# 2. DETECTION SECTION WITH CHUNKING
import whisper
import json

def detect_audio_language_and_transcribe_with_chunks(audio_path, min_pause_duration=1.5):
    """
    Detect language and transcribe audio with timestamp chunks for accurate synchronization.
    Only creates new chunks when pauses are longer than min_pause_duration.
    
    Args:
        audio_path: Path to the audio file
        min_pause_duration: Minimum pause duration in seconds to create a new chunk (default: 1.5)
    """
    print(f" Detecting language and transcribing with timestamps (min pause: {min_pause_duration}s)...")
    model = whisper.load_model("small")

    # First detect language only
    audio = whisper.load_audio(audio_path)
    audio_sample = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_sample).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f" Detected Language: {detected_language} (confidence: {probs[detected_language]:.2f})")

    # Now transcribe using the original audio with word-level timestamps
    print(" Transcribing audio with timestamps...")
    result = model.transcribe(audio_path, word_timestamps=True)

    full_text = result["text"]
    print(f" Original Full Text: {full_text}")

    # Extract segments and merge those with short pauses
    segments = result["segments"]
    print(f" Whisper detected {len(segments)} segments initially")
    
    # Merge segments that have pauses shorter than min_pause_duration
    merged_chunks = []
    current_chunk = None
    
    for i, segment in enumerate(segments):
        if current_chunk is None:
            # Start a new chunk
            current_chunk = {
                "start": segment["start"],
                "end": segment["end"],
                "original_text": segment["text"].strip(),
                "translated_text": None
            }
        else:
            # Check pause duration between current chunk end and this segment start
            pause_duration = segment["start"] - current_chunk["end"]
            
            if pause_duration >= min_pause_duration:
                # Long pause detected - save current chunk and start new one
                merged_chunks.append(current_chunk)
                print(f"   ⏸ Long pause detected: {pause_duration:.2f}s at {current_chunk['end']:.2f}s")
                current_chunk = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "original_text": segment["text"].strip(),
                    "translated_text": None
                }
            else:
                # Short pause - merge with current chunk
                current_chunk["end"] = segment["end"]
                current_chunk["original_text"] += " " + segment["text"].strip()
    
    # Don't forget to add the last chunk
    if current_chunk is not None:
        merged_chunks.append(current_chunk)
    
    chunks = merged_chunks
    print(f"✅ Created {len(chunks)} chunks after merging (pauses < {min_pause_duration}s ignored)")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        duration = chunk["end"] - chunk["start"]
        print(f"   Chunk {i+1}: [{chunk['start']:.2f}s - {chunk['end']:.2f}s] ({duration:.2f}s) '{chunk['original_text'][:50]}...'")
    
    return full_text, detected_language, chunks

def detect_audio_language_and_transcribe(audio_path):
    """Detect language and transcribe audio - with optional slowing for non-English"""
    print("🔍 Detecting language...")
    model = whisper.load_model("small")

    # First detect language only
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"🎯 Detected Language: {detected_language} (confidence: {probs[detected_language]:.2f})")

    # Now transcribe using the original audio
    print("📝 Transcribing audio...")
    result = model.transcribe(audio_path)

    print("Original Text:", result["text"])
    return result["text"], detected_language

def detect_audio_language(audio_path):
    """Wrapper function for backward compatibility"""
    text, language = detect_audio_language_and_transcribe(audio_path)
    return text

# 3. DEEP-TRANSLATOR TRANSLATION SECTION
from deep_translator import GoogleTranslator

def optimize_text_for_timing(text, target_lang, gemini_api_key, original_duration):
    """Use Sarvam to optimize text length to fit the original video duration"""
    print("🧠 Using Sarvam to optimize text for perfect timing...")

    try:
        # Initialize the client with API key
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to Gemini 2.0


        # Language mapping for the prompt
        lang_names = {
            'hi': 'Hindi', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu',
            'bn': 'Bengali', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'en': 'English', 'pa': 'Punjabi'
        }

        lang_name = lang_names.get(target_lang, 'the target language')

        # Estimate words per second for natural speech (adjust based on language)
        words_per_second = {
            'hi': 2.5, 'mr': 2.5, 'ta': 2.8, 'te': 2.8, 'bn': 2.6,
            'es': 3.0, 'fr': 3.0, 'de': 3.0, 'it': 3.0, 'en': 3.0, 'pa': 2.5
        }

        target_wps = words_per_second.get(target_lang, 2.8)
        target_word_count = int(original_duration * target_wps)

        prompt = f"""
        You are an experienced teacher explaining concepts to students. Rewrite the following text in clear, educational {lang_name}.

        CRITICAL TIMING CONSTRAINT:
        - The final spoken version MUST fit within {original_duration:.1f} seconds
        - Target word count: approximately {target_word_count} words
        - Speak at a natural teaching pace (about {target_wps} words per second)

        TEACHING GUIDELINES:
        - Explain concepts clearly like a teacher to students
        - Convert the normal english terms back to english
        - Use proper educational language but keep it accessible
        - Maintain academic accuracy while being engaging
        - Structure the explanation logically
        - Adjust the length to fit the time constraint perfectly
        - Keep the core meaning exactly the same
        - Make sure that the text you provide is completely in the given language even words in english should be transliterated in hindi text and does not have any english text even in brackets and do not put asteriks.

        IMPORTANT SPEECH CONSIDERATION:
        - The text you provide will be converted into speech, so ensure it flows naturally when spoken
        - Use language that sounds natural and conversational when read aloud
        - Avoid complex sentence structures that might be difficult to pronounce
        - Use punctuation that guides natural pauses and intonation

        LENGTH ADJUSTMENT STRATEGY:
        If the text is too long: Remove redundant parts, use more concise language
        If the text is too short: Add brief explanations, examples, or context
        Aim for natural pacing that fits {original_duration:.1f} seconds exactly

        TEXT TO OPTIMIZE: "{text}"

        Return only the timing-optimized educational version without any additional explanations.
        The spoken version of your response must naturally take {original_duration:.1f} seconds to deliver.
        """
        response = model.generate_content(prompt)
        optimized_text = response.text.strip()

        # Clean up the response
        if optimized_text.startswith('"') and optimized_text.endswith('"'):
            optimized_text = optimized_text[1:-1]

        remove_phrases = [
            "Here's the timing-optimized version:", "As a teacher would explain within the time limit:",
            "Educational version optimized for timing:", "Here's the teacher's explanation fitting the duration:",
            "Sure, here's the timing-optimized educational rewrite:", "Of course! Here's the version that fits the time constraint:"
        ]
        for phrase in remove_phrases:
            if optimized_text.startswith(phrase):
                optimized_text = optimized_text[len(phrase):].strip()

        actual_word_count = len(optimized_text.split())
        print(f"✅ Teacher-Optimized Text ({actual_word_count} words): {optimized_text}")
        print(f"⏱ Target: {target_word_count} words for {original_duration:.1f}s duration")
        return optimized_text

    except Exception as e:
        print(f"❌ Gemini optimization failed: {e}")
        print("🔄 Using direct translation instead...")
        return GoogleTranslator(source='auto', target=target_lang).translate(text)

def map_translation_to_chunks(full_translated_text, chunks):
    """Map the full translated text back to individual chunks proportionally"""
    print("🔄 Mapping translated text to chunks...")
    
    # Split translated text into words
    translated_words = full_translated_text.split()
    
    # Calculate total original word count
    total_original_words = sum(len(chunk["original_text"].split()) for chunk in chunks)
    
    if total_original_words == 0:
        print("⚠️  No original words found, using equal distribution")
        words_per_chunk = len(translated_words) // len(chunks) if len(chunks) > 0 else 0
        
        word_idx = 0
        for chunk in chunks:
            chunk_words = translated_words[word_idx:word_idx + words_per_chunk]
            chunk["translated_text"] = " ".join(chunk_words)
            word_idx += words_per_chunk
        
        # Add remaining words to last chunk
        if word_idx < len(translated_words):
            chunks[-1]["translated_text"] += " " + " ".join(translated_words[word_idx:])
    else:
        # Distribute words proportionally based on original text length
        word_idx = 0
        for i, chunk in enumerate(chunks):
            original_word_count = len(chunk["original_text"].split())
            proportion = original_word_count / total_original_words
            
            # Calculate how many translated words this chunk should get
            chunk_word_count = max(1, round(len(translated_words) * proportion))
            
            # For last chunk, take all remaining words
            if i == len(chunks) - 1:
                chunk_words = translated_words[word_idx:]
            else:
                chunk_words = translated_words[word_idx:word_idx + chunk_word_count]
            
            chunk["translated_text"] = " ".join(chunk_words)
            word_idx += len(chunk_words)
    
    # Verify all chunks have translated text
    for i, chunk in enumerate(chunks):
        if not chunk["translated_text"] or chunk["translated_text"].strip() == "":
            print(f"⚠️  Chunk {i+1} has empty translation, using original")
            chunk["translated_text"] = chunk["original_text"]
        print(f"   Chunk {i+1}: '{chunk['translated_text'][:50]}...'")
    
    print(f"✅ Mapped translation to {len(chunks)} chunks")
    return chunks

def translate_with_timing_optimization(text, target_lang, gemini_api_key, original_duration):
    """Translate text with Gemini educational optimization and timing constraint"""
    print("🌍 Translating with timing optimization...")

    try:
        print(f"Translating from detected language to {target_lang}...")
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        print(f"Initial Translation: {translated}")

        if len(translated.split()) > 3:
            optimized = optimize_text_for_timing(translated, target_lang, gemini_api_key, original_duration)
            return optimized
        else:
            print("📝 Text too short for optimization, using direct translation")
            return translated

    except Exception as e:
        print(f"❌ Enhanced translation failed: {e}")
        return GoogleTranslator(source='auto', target=target_lang).translate(text)


# 4. VOICE SECTION (Svara TTS)

# --------------------------
# Svara TTS Model Loading
# --------------------------
print("⚙ Loading Svara TTS models...")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔩 Using device: {device}")

model_name = "kenpath/svara-tts-v1"

try:
    print("📥 Loading SNAC model (this may take a moment)...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    print("✅ SNAC model loaded!")
    
    print("📥 Loading Svara TTS model (this is a large model, please wait)...")
    print("💡 TIP: If this fails with memory errors, try:")
    print("   1. Close other applications to free up RAM")
    print("   2. Increase Windows virtual memory (paging file size)")
    print("   3. Or use a smaller TTS model")
    
    # Load with memory-efficient settings
    svara_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision to save memory
        low_cpu_mem_usage=True,      # Optimize CPU memory usage
        device_map="auto"             # Automatically map to GPU
    )
    svara_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Svara TTS models loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load Svara TTS models: {e}")
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Increase Windows Virtual Memory:")
    print("   - Settings > System > About > Advanced system settings")
    print("   - Advanced tab > Performance Settings > Advanced")
    print("   - Virtual memory > Change > Set to 16GB+ (16384 MB)")
    print("\n2. Close memory-intensive applications")
    print("\n3. Restart your computer and try again")
    print("\n⚠️  Cannot proceed without TTS model. Exiting...")
    import sys
    sys.exit(1)
# --------------------------

def generate_audio_from_text(text, language, gender, emotion="Happy"):
    """
    Generate audio from text using the Svara-TTS model with emotion control.
    
    Args:
        text: Text to convert to speech
        language: Language for TTS (e.g., 'Hindi', 'English')
        gender: Voice gender ('Male' or 'Female')
        emotion: Emotion to apply ('Happy', 'Sad', 'Angry', 'Neutral')
    """
    # Format with emotion for consistent and expressive voice
    voice = f"{language} ({gender}) [{emotion}]"
    formatted_text = f"<|audio|> {voice}: {text}<|eot_id|>"
    prompt = "<custom_token_3>" + formatted_text + "<custom_token_4><custom_token_5>"

    input_ids = svara_tokenizer(prompt, return_tensors="pt").input_ids
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    input_ids = modified_input_ids.to(device)

    with torch.no_grad():
        generated_ids = svara_model.generate(
            input_ids=input_ids, 
            max_new_tokens=5000,      # Increased from 800 to 5000 for 180-second audio
            do_sample=True, 
            temperature=0.7,          # Consistent temperature for voice stability
            top_p=0.95, 
            repetition_penalty=1.2, 
            num_return_sequences=1, 
            eos_token_id=128258,
        )

    START_OF_SPEECH_TOKEN, END_OF_SPEECH_TOKEN = 128257, 128258
    AUDIO_CODE_BASE_OFFSET, AUDIO_CODE_MAX = 128266, 128266 + (7 * 4096) - 1
    row = generated_ids[0]
    token_indices = (row == START_OF_SPEECH_TOKEN).nonzero(as_tuple=True)[0]

    if len(token_indices) > 0:
        start_idx = token_indices[-1].item() + 1
        audio_tokens = row[start_idx:]
        audio_tokens = audio_tokens[audio_tokens != END_OF_SPEECH_TOKEN]
        audio_tokens = audio_tokens[audio_tokens != 128263]
        valid_mask = (audio_tokens >= AUDIO_CODE_BASE_OFFSET) & (audio_tokens <= AUDIO_CODE_MAX)
        audio_tokens = audio_tokens[valid_mask]
        snac_tokens = [t - AUDIO_CODE_BASE_OFFSET for t in audio_tokens.tolist()]
        new_length = (len(snac_tokens) // 7) * 7
        snac_tokens = snac_tokens[:new_length]
    else:
        raise ValueError("No speech tokens found in generated output")

    def redistribute_codes(code_list):
        codes_lvl = [[] for _ in range(3)]
        llm_codebook_offsets = [i * 4096 for i in range(7)]
        for i in range(0, len(code_list), 7):
            codes_lvl[0].append(code_list[i] - llm_codebook_offsets[0])
            codes_lvl[1].extend([code_list[i+1] - llm_codebook_offsets[1], code_list[i+4] - llm_codebook_offsets[4]])
            codes_lvl[2].extend([
                code_list[i+2] - llm_codebook_offsets[2], code_list[i+3] - llm_codebook_offsets[3],
                code_list[i+5] - llm_codebook_offsets[5], code_list[i+6] - llm_codebook_offsets[6]
            ])
        hierarchical_codes = [torch.tensor(lvl_codes, dtype=torch.long, device=device).unsqueeze(0) for lvl_codes in codes_lvl]
        with torch.no_grad():
            return snac_model.decode(hierarchical_codes)

    audio_waveform = redistribute_codes(snac_tokens)
    return audio_waveform.detach().squeeze().to("cpu").numpy()

def text_to_speech_svara(text, target_lang_code, output_filename="translated_audio.wav", gender="Male", emotion="Happy"):
    """Convert text to speech using Svara TTS with emotion and save it to a file."""
    print(f"🔊 Converting text to speech with Svara TTS (Emotion: {emotion})...")
    
    # Map language codes to full names required by Svara TTS
    svara_language_map = {
        'hi': 'Hindi', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu',
        'bn': 'Bengali', 'es': 'English', 'fr': 'English', 'de': 'English',
        'it': 'English', 'en': 'English', 'gu': 'Gujarati', 'kn': 'Kannada',
        'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Hindi'
    }
    
    language = svara_language_map.get(target_lang_code, 'English')
    print(f"🗣 Using Svara TTS with language: {language} ({gender}) [{emotion}]")

    try:
        audio_array = generate_audio_from_text(text, language, gender, emotion)
        sample_rate = 24000  # Svara TTS sample rate
        sf.write(output_filename, audio_array, sample_rate)
        
        # Display audio for notebook environments
        display(Audio(audio_array, rate=sample_rate, autoplay=False))
        
        print(f"✅ Audio generated successfully and saved to {output_filename}!")
        return output_filename
    except Exception as e:
        print(f"❌ Svara TTS audio generation failed: {e}")
        raise

def generate_chunk_audio(chunk, chunk_index, target_lang_code, gender="Male", emotion="Happy"):
    """Generate audio for a single chunk with consistent voice and emotion"""
    chunk_text = chunk["translated_text"]
    chunk_duration = chunk["end"] - chunk["start"]
    
    print(f"🎙 Chunk {chunk_index}: [{chunk['start']:.2f}s - {chunk['end']:.2f}s] ({chunk_duration:.2f}s)")
    print(f"   Text: '{chunk_text[:60]}...'")
    
    # Map language codes to full names required by Svara TTS
    svara_language_map = {
        'hi': 'Hindi', 'mr': 'Marathi', 'ta': 'Tamil', 'te': 'Telugu',
        'bn': 'Bengali', 'es': 'English', 'fr': 'English', 'de': 'English',
        'it': 'English', 'en': 'English', 'gu': 'Gujarati', 'kn': 'Kannada',
        'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Hindi'
    }
    
    language = svara_language_map.get(target_lang_code, 'English')
    
    try:
        # Generate audio array with consistent voice parameters (same gender and emotion for all chunks)
        audio_array = generate_audio_from_text(chunk_text, language, gender, emotion)
        sample_rate = 24000  # Svara TTS sample rate
        
        # Calculate actual audio duration
        audio_duration = len(audio_array) / sample_rate
        
        # Speed adjustment logic
        if audio_duration > chunk_duration:
            # Audio is longer than chunk duration - speed it up
            speed_factor = audio_duration / chunk_duration
            print(f"   ⚡ Audio too long ({audio_duration:.2f}s), speeding up by {speed_factor:.2f}x")
            
            # Use librosa for time-stretching (speed change without pitch change)
            import librosa
            audio_array = librosa.effects.time_stretch(audio_array, rate=speed_factor)
            audio_duration = len(audio_array) / sample_rate
        
        print(f"   ✅ Generated audio: {audio_duration:.2f}s")
        
        chunk["audio_array"] = audio_array
        chunk["audio_duration"] = audio_duration
        chunk["sample_rate"] = sample_rate
        
        return chunk
        
    except Exception as e:
        print(f"   ❌ Failed to generate audio for chunk {chunk_index}: {e}")
        chunk["audio_array"] = None
        chunk["audio_duration"] = 0
        chunk["sample_rate"] = 24000
        return chunk

def generate_chunk_audio_parallel(chunk_data):
    """
    Wrapper function for parallel chunk audio generation.
    
    Args:
        chunk_data: Tuple of (chunk, chunk_index, target_lang_code, gender, emotion)
    
    Returns:
        Tuple of (chunk_index, updated_chunk)
    """
    chunk, chunk_index, target_lang_code, gender, emotion = chunk_data
    
    try:
        updated_chunk = generate_chunk_audio(chunk, chunk_index, target_lang_code, gender, emotion)
        return (chunk_index, updated_chunk)
    except Exception as e:
        print(f"   ❌ Parallel processing failed for chunk {chunk_index}: {e}")
        chunk["audio_array"] = None
        chunk["audio_duration"] = 0
        chunk["sample_rate"] = 24000
        return (chunk_index, chunk)

def generate_all_chunks_parallel(chunks, target_lang_code, gender="Male", emotion="Happy", max_workers=None):
    """
    Generate TTS audio for all chunks in parallel using ThreadPoolExecutor.
    
    Args:
        chunks: List of chunk dictionaries
        target_lang_code: Target language code
        gender: Voice gender
        emotion: Voice emotion
        max_workers: Maximum number of parallel workers (default: min(CPU count, chunk count))
    
    Returns:
        Updated chunks list with audio data
    """
    print(f"\n🚀 Generating TTS audio for {len(chunks)} chunks IN PARALLEL...")
    
    # Determine optimal number of workers
    if max_workers is None:
        import multiprocessing
        max_workers = min(len(chunks), multiprocessing.cpu_count())
    
    print(f"⚡ Using {max_workers} parallel workers for TTS generation")
    
    # Prepare chunk data for parallel processing
    chunk_data_list = [
        (chunks[i], i+1, target_lang_code, gender, emotion) 
        for i in range(len(chunks))
    ]
    
    # Process chunks in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(generate_chunk_audio_parallel, chunk_data): chunk_data[1]
            for chunk_data in chunk_data_list
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                print(f"   ✅ Completed chunk {completed_count}/{len(chunks)}")
            except Exception as e:
                print(f"   ❌ Chunk {chunk_index} failed: {e}")
    
    # Sort results by chunk index to maintain order
    results.sort(key=lambda x: x[0])
    
    # Update chunks with processed audio data
    updated_chunks = []
    for chunk_index, updated_chunk in results:
        updated_chunks.append(updated_chunk)
    
    print(f"✅ Parallel TTS generation complete for all {len(chunks)} chunks!")
    return updated_chunks

def assemble_chunk_audios_to_video(video_path, chunks, output_path):
    """Assemble chunk audios into a single audio track aligned with video timestamps"""
    print("🎬 Assembling chunk audios into final video...")
    
    # Get video duration
    video = VideoFileClip(video_path)
    video_duration = video.duration
    sample_rate = 24000  # Svara TTS sample rate
    
    # Create silent audio array for the full video duration
    total_samples = int(video_duration * sample_rate)
    final_audio_array = np.zeros(total_samples, dtype=np.float32)
    
    print(f"📊 Video duration: {video_duration:.2f}s, Total samples: {total_samples}")
    
    # Place each chunk audio at its timestamp position
    for i, chunk in enumerate(chunks):
        if chunk["audio_array"] is None:
            print(f"⚠️  Skipping chunk {i+1} (no audio)")
            continue
        
        # Calculate sample positions
        start_sample = int(chunk["start"] * sample_rate)
        chunk_audio = chunk["audio_array"]
        chunk_samples = len(chunk_audio)
        
        # Ensure we don't exceed the final audio array bounds
        end_sample = min(start_sample + chunk_samples, total_samples)
        actual_chunk_samples = end_sample - start_sample
        
        # Place the chunk audio at the correct timestamp
        final_audio_array[start_sample:end_sample] = chunk_audio[:actual_chunk_samples]
        
        print(f"   ✅ Placed chunk {i+1} at {chunk['start']:.2f}s ({actual_chunk_samples} samples)")
    
    # Save the assembled audio to a temporary file
    temp_audio_path = "assembled_chunk_audio.wav"
    sf.write(temp_audio_path, final_audio_array, sample_rate)
    print(f"✅ Assembled audio saved to {temp_audio_path}")
    
    # Merge with video
    print("🎬 Merging assembled audio with video...")
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', temp_audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Video with chunk-based audio created successfully!")
        
        # Cleanup
        video.close()
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg merge failed: {e}")
        print("🔄 Falling back to MoviePy...")
        
        # Fallback to MoviePy
        new_audio = AudioFileClip(temp_audio_path)
        final_video = video.set_audio(new_audio)
        final_video.write_videofile(output_path, verbose=False, logger=None, codec='libx264')
        
        video.close()
        new_audio.close()
        final_video.close()
        
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print("✅ Video created (MoviePy fallback)!")
        return output_path

# MAIN PIPELINE WITH YOUTUBE DOWNLOAD AND TRIMMING
def youtube_video_translation_pipeline(youtube_url, target_language, gemini_api_key):
    """Complete pipeline: YouTube URL → Download → Trim → Audio → Detect → Translate → Voice (Svara) → Adjust → Merge"""
    print("🚀 Starting YouTube video translation pipeline with Svara TTS...")

    downloaded_video_path = download_youtube_video(youtube_url, "downloaded_video.mp4")
    trimmed_video_path = trim_video_to_first_minute(downloaded_video_path, "trimmed_video.mp4")
    original_duration = get_video_duration(trimmed_video_path)
    print(f"⏱ Trimmed video duration: {original_duration:.2f} seconds")

    extracted_audio_path = extract_audio_from_video(trimmed_video_path)
    original_text = detect_audio_language(extracted_audio_path)
    translated_text = translate_with_timing_optimization(original_text, target_language, gemini_api_key, original_duration)
    
    # Use Svara TTS for speech synthesis
    new_audio_path = text_to_speech_svara(translated_text, target_language, "translated_audio.wav")

    new_audio_duration = AudioFileClip(new_audio_path).duration
    AudioFileClip(new_audio_path).close()
    print(f"⏱ Generated audio duration: {new_audio_duration:.2f} seconds")
    print(f"⏱ Original video duration: {original_duration:.2f} seconds")

    adjusted_video_path = "adjusted_video.mp4"
    adjust_video_speed(trimmed_video_path, original_duration, new_audio_duration, adjusted_video_path)

    output_video_path = "final_translated_video.mp4"
    final_video = merge_audio_with_video(adjusted_video_path, new_audio_path, output_video_path)

    # Cleanup with retry logic to handle file locking
    print("🧹 Cleaning up temporary files...")
    for temp_file in [downloaded_video_path, trimmed_video_path, extracted_audio_path, new_audio_path, adjusted_video_path]:
        if os.path.exists(temp_file):
            try:
                time.sleep(0.5)  # Give time for file handles to release
                os.remove(temp_file)
                print(f"✅ Deleted: {temp_file}")
            except PermissionError:
                print(f"⚠️  Could not delete {temp_file} (file in use, will be cleaned up later)")
            except Exception as e:
                print(f"⚠️  Could not delete {temp_file}: {e}")

    print("🎉 Pipeline complete! Final video with Svara TTS audio ready!")
    return final_video

def youtube_video_translation_pipeline_chunked(youtube_url, target_language, gemini_api_key, gender="Male", emotion="Happy", min_pause_duration=1.5, parallel_tts=True):
    """
    NEW CHUNK-BASED PIPELINE with timestamp-accurate audio synchronization and emotion control.
    
    Process:
    1. Download and trim video
    2. Extract audio
    3. Transcribe with Whisper (creates chunks with timestamps, only splits on pauses >= min_pause_duration)
    4. Translate full text with Gemini optimization
    5. Map translated text to chunks proportionally
    6. Generate TTS audio for each chunk with consistent voice and emotion (PARALLEL or SEQUENTIAL)
    7. Adjust chunk audio speed if needed
    8. Assemble chunks at correct timestamps
    9. Merge with video
    
    Args:
        youtube_url: YouTube video URL
        target_language: Target language code (e.g., 'bn', 'hi')
        gemini_api_key: Gemini API key for translation optimization
        gender: Voice gender ('Male' or 'Female') - kept consistent across all chunks
        emotion: Voice emotion ('Happy', 'Sad', 'Angry', 'Neutral') - kept consistent across all chunks
        min_pause_duration: Minimum pause duration (seconds) to create new chunk (default: 1.5)
        parallel_tts: If True, process TTS for chunks in parallel for faster generation (default: True)
    """
    print(f" Starting CHUNK-BASED YouTube video translation pipeline with Svara TTS...")
    print(f" Voice Settings: {gender} voice with {emotion} emotion (consistent across all chunks)")
    print(f"⏸ Pause Detection: Only pauses ≥ {min_pause_duration}s will create new chunks")
    print(f" TTS Processing Mode: {'PARALLEL (FASTER!)' if parallel_tts else 'SEQUENTIAL'}")

    # 1. Download and prepare video
    downloaded_video_path = download_youtube_video(youtube_url, "downloaded_video.mp4")
    trimmed_video_path = trim_video_to_first_minute(downloaded_video_path, "trimmed_video.mp4")
    original_duration = get_video_duration(trimmed_video_path)
    print(f" Trimmed video duration: {original_duration:.2f} seconds")

    # 2. Extract audio
    extracted_audio_path = extract_audio_from_video(trimmed_video_path)
    
    # 3. Transcribe with chunks and timestamps (only split on pauses >= min_pause_duration)
    full_original_text, detected_language, chunks = detect_audio_language_and_transcribe_with_chunks(
        extracted_audio_path, 
        min_pause_duration=min_pause_duration
    )
    
    # 4. Translate full text with Gemini optimization
    full_translated_text = translate_with_timing_optimization(
        full_original_text, 
        target_language, 
        gemini_api_key, 
        original_duration
    )
    
    # 5. Map translated text to chunks
    chunks = map_translation_to_chunks(full_translated_text, chunks)
    
    # 6. Generate TTS audio for each chunk (PARALLEL or SEQUENTIAL)
    if parallel_tts and len(chunks) > 1:
        # Use parallel processing for multiple chunks (MUCH FASTER!)
        chunks = generate_all_chunks_parallel(chunks, target_language, gender, emotion)
    else:
        # Use sequential processing (fallback or single chunk)
        print(f"\n🎙 Generating TTS audio for {len(chunks)} chunks sequentially...")
        for i in range(len(chunks)):
            chunks[i] = generate_chunk_audio(chunks[i], i+1, target_language, gender, emotion)
    
    # 7 & 8. Assemble chunks and merge with video
    output_video_path = "final_translated_video_chunked.mp4"
    final_video = assemble_chunk_audios_to_video(trimmed_video_path, chunks, output_video_path)

    # Cleanup
    print(" Cleaning up temporary files...")
    for temp_file in [downloaded_video_path, trimmed_video_path, extracted_audio_path]:
        if os.path.exists(temp_file):
            try:
                time.sleep(0.5)
                os.remove(temp_file)
                print(f"✅ Deleted: {temp_file}")
            except PermissionError:
                print(f"⚠️  Could not delete {temp_file} (file in use)")
            except Exception as e:
                print(f"⚠️  Could not delete {temp_file}: {e}")

    print(" CHUNK-BASED Pipeline complete! Final video with timestamp-accurate audio ready!")
    return final_video

# SIMPLE VERSION WITHOUT GEMINI (Only deep-translator)
def simple_youtube_video_translation_pipeline(youtube_url, target_language):
    """Simplified pipeline using Svara TTS and without Gemini optimization."""
    print(" Starting simple YouTube video translation pipeline with Svara TTS...")

    downloaded_video_path = download_youtube_video(youtube_url, "downloaded_video.mp4")
    trimmed_video_path = trim_video_to_first_minute(downloaded_video_path, "trimmed_video.mp4")
    original_duration = get_video_duration(trimmed_video_path)
    print(f"⏱ Trimmed video duration: {original_duration:.2f} seconds")

    extracted_audio_path = extract_audio_from_video(trimmed_video_path)
    original_text = detect_audio_language(extracted_audio_path)
    
    print(f" Translating to {target_language}...")
    translated_text = GoogleTranslator(source='auto', target=target_language).translate(original_text)
    print(f" Translated Text: {translated_text}")

    # Use Svara TTS for speech synthesis
    new_audio_path = text_to_speech_svara(translated_text, target_language, "translated_audio.wav")
    
    new_audio_duration = AudioFileClip(new_audio_path).duration
    AudioFileClip(new_audio_path).close()
    print(f"⏱ Generated audio duration: {new_audio_duration:.2f} seconds")

    adjusted_video_path = "adjusted_video.mp4"
    adjust_video_speed(trimmed_video_path, original_duration, new_audio_duration, adjusted_video_path)

    output_video_path = "final_translated_video.mp4"
    final_video = merge_audio_with_video(adjusted_video_path, new_audio_path, output_video_path)

    # Cleanup with retry logic to handle file locking
    print(" Cleaning up temporary files...")
    for temp_file in [downloaded_video_path, trimmed_video_path, extracted_audio_path, new_audio_path, adjusted_video_path]:
        if os.path.exists(temp_file):
            try:
                time.sleep(0.5)  # Give time for file handles to release
                os.remove(temp_file)
                print(f"✅ Deleted: {temp_file}")
            except PermissionError:
                print(f"⚠️  Could not delete {temp_file} (file in use, will be cleaned up later)")
            except Exception as e:
                print(f"⚠️  Could not delete {temp_file}: {e}")

    print(" Simple pipeline complete! Final video with Svara TTS audio ready!")
    return final_video


# USAGE
# Note: The ELEVENLABS_API_KEY is no longer needed.
GEMINI_API_KEY = "gemini_API_KEY_HERE"  # Replace with your actual Gemini API key

# Available language codes for deep-translator:
LANGUAGE_CODES = {
    'Hindi': 'hi', 'Marathi': 'mr', 'Tamil': 'ta', 'Telugu': 'te',
    'Bengali': 'bn', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
    'Italian': 'it', 'English': 'en', 'Gujarati': 'gu', 'Kannada': 'kn',
    'Malayalam': 'ml', 'Punjabi': 'pa', 'Urdu': 'ur'
}

print("Available Languages:")
for lang, code in LANGUAGE_CODES.items():
    print(f"  {lang}: '{code}'")

# Run the pipeline with YouTube URL
youtube_url = "Youtube_URL_Here"  # Replace with your YouTube URL
target_lang = "hi"  # Replace with your target language code

# NEW: Chunk-based pipeline with timestamp-accurate audio and emotion control (RECOMMENDED)
print("\n" + "="*70)
print(" USING CHUNK-BASED PIPELINE (Parallel TTS + Timestamp-Accurate Audio)")
print("="*70 + "\n")
final_video = youtube_video_translation_pipeline_chunked(
    youtube_url, 
    target_lang, 
    GEMINI_API_KEY, 
    gender="Male",          # Voice gender (consistent across all chunks)
    emotion="Happy",        # Voice emotion: 'Happy', 'Sad', 'Angry', 'Neutral' (consistent across all chunks)
    min_pause_duration=1.5, # Only create new chunks for pauses >= 1.5 seconds (merges short pauses)
    parallel_tts=True       # Enable parallel TTS processing (MUCH FASTER! 3-5x speedup)
)

# OLD: Full pipeline with Gemini and Svara TTS (single audio file)
# final_video = youtube_video_translation_pipeline(youtube_url, target_lang, GEMINI_API_KEY)

# To run the simpler version without Gemini optimization:
# final_video = simple_youtube_video_translation_pipeline(youtube_url, target_lang)


# Download the final video (This works in Google Colab)
try:
    from google.colab import files
    files.download(final_video)
except ImportError:
    print(f"Process completed successfully! Find your video at: {final_video}")