import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import os
from dotenv import load_dotenv
from typing import List, Dict
import datetime
import soundfile as sf
import time

load_dotenv('./.env')


def seconds_to_hms(seconds):
    """
    Converte segundos para horas, minutos e segundos
    :param seconds: int ou float
    :return: str
    """
    if seconds is None or not isinstance(seconds, (int, float)):
        return "---"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


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
        os.getenv("WHISPER_VERSAO"),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(os.getenv("WHISPER_VERSAO"))

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        generate_kwargs={"language": "portuguese"},
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

        os.remove(temp_path)

    return transcribed_segments


def process_audio(audio_path, hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN")
    )
    diarization = diarization.to(torch.device(device))

    audio, sr = librosa.load(audio_path, sr=16000)

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

    transcribed_segments = segment_audio_by_speakers(audio_path, speakers_segments)

    return transcribed_segments


if __name__ == "__main__":
    tempo_inicio = time.time()

    try:
        results = process_audio(
            "./audios/noticia.mp3",
            os.getenv("HF_TOKEN")
        )

        print("\nTranscrição com timestamps e falantes:")
        print("=====================================\n")

        for segment in results:
            timestamp = format_timestamp(segment["start"])
            speaker = segment["speaker"].replace("SPEAKER_", "Falante ")
            text = segment["text"]

            print(f"[{timestamp}] ({speaker}) - {text}")

        tempo_fim = time.time()
        tempo_total = tempo_fim - tempo_inicio
        tempo_total = time.strftime("%H:%M:%S", time.gmtime(tempo_total))

        print(f"\n\n{tempo_total}")

    except Exception as e:
        print(f"Erro ao processar o áudio: {str(e)}")
