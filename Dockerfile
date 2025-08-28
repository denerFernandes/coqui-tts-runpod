FROM python:3.10-slim

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN echo "y" | python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" || true
COPY handler.py /

# Start the container
CMD ["python3", "-u", "handler.py"]