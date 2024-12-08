import torch
import numpy as np
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import os
from dotenv import load_dotenv
from typing import List, Dict
import datetime
import soundfile as sf

load_dotenv('../.env')


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=int(seconds)))


def segment_audio_by_speakers(audio_path: str, segments: List[Dict]) -> List[Dict]:
    """
    Segmenta o áudio de acordo com os intervalos dos falantes e transcreve cada segmento
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Initialize Whisper
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Carrega o áudio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    transcribed_segments = []
    
    for segment in segments:
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        
        # Extrai o segmento do áudio
        segment_audio = audio[start_sample:end_sample]
        
        # Salva temporariamente o segmento
        temp_path = f"temp_segment_{segment['start']}_{segment['end']}.wav"
        sf.write(temp_path, segment_audio, sr)
        
        # Transcreve o segmento
        result = pipe(temp_path)
        
        # Adiciona a transcrição ao dicionário do segmento
        segment["text"] = result["text"].strip()
        transcribed_segments.append(segment)
        
        # Remove o arquivo temporário
        import os
        os.remove(temp_path)
    
    return transcribed_segments

def process_audio(audio_path, hf_token):
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Pyannote
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization = diarization.to(torch.device(device))

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)

    # Perform diarization
    diarization_result = diarization({
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": sr
    })

    # Process segments
    speakers_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time = round(turn.start, 2)
        end_time = round(turn.end, 2)
        speakers_segments.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time
        })

    # Segment and transcribe audio
    transcribed_segments = segment_audio_by_speakers(audio_path, speakers_segments)
    
    return transcribed_segments

if __name__ == "__main__":
    # You need to replace this with your Hugging Face token
    HF_TOKEN = os.getenv("HF_TOKEN")
    AUDIO_PATH = "./noticia.mp3"
    
    try:
        results = process_audio(AUDIO_PATH, HF_TOKEN)
        
        print("\nTranscrição com timestamps e falantes:")
        print("=====================================\n")
        
        for segment in results:
            timestamp = format_timestamp(segment["start"])
            speaker = segment["speaker"].replace("SPEAKER_", "Falante ")
            text = segment["text"]
            
            print(f"[{timestamp}] ({speaker}) - {text}")
            
    except Exception as e:
        print(f"Erro ao processar o áudio: {str(e)}")
