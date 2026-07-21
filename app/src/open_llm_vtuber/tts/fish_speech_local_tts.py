import asyncio
import io
import os
import tempfile
import wave
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """Streaming adapter for a local Fish Speech 1.5 API server."""

    supports_streaming = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:50000",
        reference_id: str = "olv_263_oral5",
        sample_rate: int = 44100,
        chunk_length: int = 100,
        chunk_seconds: float = 0.8,
        seed: int = 42,
        top_p: float = 0.6,
        repetition_penalty: float = 1.2,
        temperature: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.reference_id = reference_id
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.chunk_seconds = chunk_seconds
        self.seed = seed
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self._request_lock = asyncio.Lock()

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        raise NotImplementedError("Fish Speech is configured as streaming-only TTS")

    def _write_pcm_chunk(self, pcm: bytes) -> str:
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)

        os.makedirs("cache", exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".wav",
            prefix="fish_speech_",
            dir="cache",
            delete=False,
        ) as output:
            output.write(output_buffer.getvalue())
            return output.name

    async def async_stream_audio(self, text: str) -> AsyncIterator[str]:
        request = {
            "text": text,
            "chunk_length": self.chunk_length,
            "format": "wav",
            "reference_id": self.reference_id,
            "use_memory_cache": "on",
            "streaming": True,
            "normalize": True,
            "max_new_tokens": 1024,
            "seed": self.seed,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
        }
        # Fish Speech 1.5 streams headerless mono int16 PCM even though the
        # endpoint content type is audio/wav. Repackage fixed-size PCM groups
        # into independent WAV files for the browser audio queue.
        group_bytes = max(
            2,
            int(self.sample_rate * 2 * self.chunk_seconds) // 2 * 2,
        )

        async with self._request_lock:
            timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/tts",
                    json=request,
                ) as response:
                    response.raise_for_status()
                    chunk_index = 0
                    async for pcm in response.aiter_bytes(chunk_size=group_bytes):
                        if not pcm:
                            continue
                        if len(pcm) % 2:
                            logger.warning(
                                "Dropping one incomplete Fish Speech PCM byte"
                            )
                            pcm = pcm[:-1]
                        if not pcm:
                            continue
                        path = self._write_pcm_chunk(pcm)
                        logger.debug(
                            "Fish Speech chunk {}: {:.3f}s",
                            chunk_index,
                            len(pcm) / 2 / self.sample_rate,
                        )
                        chunk_index += 1
                        yield path
