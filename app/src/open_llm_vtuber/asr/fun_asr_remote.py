import io
import wave

import httpx
import numpy as np

from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:10095/transcribe",
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=timeout)

    def transcribe_np(self, audio: np.ndarray) -> str:
        wav_bytes = self._to_wav(audio)
        response = self.client.post(
            self.endpoint,
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def _to_wav(self, audio: np.ndarray) -> bytes:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype("<i2")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.NUM_CHANNELS)
            wav_file.setsampwidth(self.SAMPLE_WIDTH)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()
