import os
import runpod
import tempfile
import torch
import logging
import time
import gc
import numpy as np
import soundfile as sf
import librosa
import requests
import subprocess

# --- PyTorch Security Fix (MUST be before TTS imports) ---
# The following is a workaround for a security feature in PyTorch 2.1+
# which prevents loading pickled files from untrusted sources.
# We explicitly trust the XttsConfig class from the coqui-ai/TTS library.
from TTS.tts.configs.xtts_config import XttsConfig
from torch.serialization import add_safe_globals
add_safe_globals([XttsConfig])

# Now import TTS after the security fix
from TTS.api import TTS

# --- Pydub Configuration ---
# Explicitly tell pydub where to find ffmpeg
from pydub import AudioSegment
AudioSegment.converter = "/usr/bin/ffmpeg"

from runpod.serverless.utils import rp_upload

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading ---

tts_model = None

def load_model():
    """Loads the TTS model into memory."""
    global tts_model
    if tts_model is not None:
        logger.info("Model is already loaded.")
        return

    logger.info("🚀 Initializing Coqui TTS Serverless Worker...")
    is_cuda_available = torch.cuda.is_available()
    logger.info(f"🔍 CUDA available: {is_cuda_available}")

    device = "cuda" if is_cuda_available else "cpu"
    if is_cuda_available:
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ Using CPU for inference.")

    try:
        logger.info("📥 Loading XTTS-v2 model...")
        start_time = time.time()
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=is_cuda_available)
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        # If the model fails to load, we can't process any jobs.
        # We'll let the error propagate to stop the worker initialization.
        raise

# Load the model when the worker starts
load_model()

# ========== AUDIO PROCESSING FUNCTIONS (from your server.py) ==========

def preprocess_reference_audio(input_path: str, output_path: str) -> bool:
    try:
        audio, orig_sr = librosa.load(input_path, sr=None, mono=False)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        audio = audio / np.max(np.abs(audio)) * 0.8
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        logger.info(f"✅ Reference audio preprocessed: {orig_sr}Hz → 16kHz mono")
        return True
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        return False

def postprocess_to_clean_audio(input_path: str, output_path: str, output_format: str = "wav") -> bool:
    try:
        audio, sr = sf.read(input_path)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85

        if output_format.lower() == "mp3":
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, audio, 44100, subtype='PCM_16')
            try:
                AudioSegment.from_wav(temp_wav).export(output_path, format="mp3", bitrate="320k")
                os.unlink(temp_wav)
                logger.info("✅ Clean audio created: 44.1kHz stereo MP3 (320kbps)")
            except Exception as e:
                logger.warning(f"⚠️ pydub MP3 conversion failed: {e}. Trying ffmpeg...")
                subprocess.run(['ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame', '-b:a', '320k', '-y', output_path], check=True, capture_output=True)
                os.unlink(temp_wav)
                logger.info("✅ Clean audio created via ffmpeg.")
        else:
            sf.write(output_path, audio, 44100, subtype='PCM_16')
            logger.info("✅ Clean audio created: 44.1kHz stereo WAV")
        return True
    except Exception as e:
        logger.error(f"❌ Error in postprocessing: {e}")
        return False

def cleanup_temp_files(file_paths: list):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove temp file {path}: {e}")

# ========== SERVERLESS HANDLER ==========

def handler(job):
    """The main handler function for the serverless worker."""
    if tts_model is None:
        return {"error": "Model is not loaded. Worker cannot process jobs."}

    job_input = job['input']
    job_id = job['id']

    # --- Input Validation & Defaults ---
    text = job_input.get('text')
    reference_audio_url = job_input.get('reference_audio_url')
    if not text or not reference_audio_url:
        return {"error": "'text' and 'reference_audio_url' are required inputs."}

    output_format = job_input.get('output_format', 'wav').lower()
    if output_format not in ["wav", "mp3"]:
        return {"error": "'output_format' must be 'wav' or 'mp3'."}

    language = job_input.get('language', 'pt')
    logger.info(f"🎵 Starting TTS job {job_id} for language '{language}' and format '{output_format}'.")

    # --- File Handling ---
    temp_dir = tempfile.gettempdir()
    ref_path = None
    processed_ref_path = None
    raw_output_path = None
    final_output_path = None
    files_to_cleanup = []

    try:
        # 1. Download reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as ref_file:
            ref_path = ref_file.name
            files_to_cleanup.append(ref_path)
            response = requests.get(reference_audio_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                ref_file.write(chunk)
        logger.info(f"Downloaded reference audio to {ref_path}")

        # 2. Pre-process reference audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as processed_ref_file:
            processed_ref_path = processed_ref_file.name
            files_to_cleanup.append(processed_ref_path)
        if not preprocess_reference_audio(ref_path, processed_ref_path):
            return {"error": "Failed to preprocess reference audio."}

        # 3. Generate raw audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as raw_output_file:
            raw_output_path = raw_output_file.name
            files_to_cleanup.append(raw_output_path)
        
        logger.info("🎤 Generating raw audio...")
        tts_model.tts_to_file(
            text=text,
            speaker_wav=processed_ref_path,
            language=language,
            file_path=raw_output_path,
            temperature=float(job_input.get('temperature', 0.75)),
            length_penalty=float(job_input.get('length_penalty', 1.0)),
            repetition_penalty=float(job_input.get('repetition_penalty', 5.0)),
            top_k=int(job_input.get('top_k', 50)),
            top_p=float(job_input.get('top_p', 0.85)),
            speed=float(job_input.get('speed', 1.0)),
            split_sentences=bool(job_input.get('enable_text_splitting', True))
        )

        # 4. Post-process for clean audio
        file_extension = "mp3" if output_format == "mp3" else "wav"
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False, dir=temp_dir) as final_file:
            final_output_path = final_file.name
            # Don't add to cleanup yet, we need to upload it first.

        if not postprocess_to_clean_audio(raw_output_path, final_output_path, output_format):
            return {"error": "Failed to postprocess the generated audio."}

        # 5. Upload the final result
        logger.info(f"Uploading final file: {final_output_path}")
        uploaded_file = rp_upload.upload_file_to_bucket(
            job_id=job_id,
            file_path=final_output_path,
        )
        files_to_cleanup.append(final_output_path) # Now add to cleanup

        return {
            "audio_url": uploaded_file['publicUrl'],
            "format": output_format
        }

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during job {job_id}: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}
    
    finally:
        # 6. Cleanup all temporary files
        logger.info(f"Cleaning up {len(files_to_cleanup)} temporary files.")
        cleanup_temp_files(files_to_cleanup)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})