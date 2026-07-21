import asyncio
from collections.abc import AsyncIterator

from gradio_client import Client, handle_file
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        client_url="http://127.0.0.1:50000/",
        mode_checkbox_group="预训练音色",
        sft_dropdown="中文女",
        prompt_text="",
        prompt_wav_upload_url="https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
        prompt_wav_record_url="https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
        instruct_text="",
        stream=False,
        seed=0,
        speed=1.0,
        api_name="/generate_audio",
        voice_cache_id="",
    ):
        # Avoid gradio_client's Unicode status marker crashing on GBK consoles.
        self.client = Client(client_url, verbose=False)

        self.mode_checkbox_group = mode_checkbox_group
        self.sft_dropdown = sft_dropdown
        self.prompt_text = prompt_text
        self.prompt_wav_upload = (
            handle_file(prompt_wav_upload_url) if prompt_wav_upload_url else None
        )
        self.prompt_wav_record = (
            handle_file(prompt_wav_record_url) if prompt_wav_record_url else None
        )
        self.instruct_text = instruct_text
        self.stream = stream
        self.seed = seed
        self.speed = speed
        self.api_name = api_name
        self.voice_cache_id = voice_cache_id
        self.supports_streaming = bool(stream)

        # A preloaded voice is already encoded by the server, so avoid uploading
        # and re-encoding the reference WAV for every synthesized sentence.
        if self.voice_cache_id:
            self.prompt_wav_upload = None
            self.prompt_wav_record = None

    def generate_audio(self, text, file_name_no_ext=None):
        if file_name_no_ext is not None:
            logger.warning(
                "Warning: customizing the temp file name with file_name_no_ext is not supported by cosyvoice2TTS and will be ignored."
            )

        result_wav_path = self.client.predict(
            tts_text=text,
            mode_checkbox_group=self.mode_checkbox_group,
            sft_dropdown=self.sft_dropdown,
            prompt_text=self.prompt_text,
            prompt_wav_upload=self.prompt_wav_upload,
            prompt_wav_record=self.prompt_wav_record,
            instruct_text=self.instruct_text,
            stream=self.stream,
            seed=self.seed,
            speed=self.speed,
            voice_cache_id=self.voice_cache_id,
            api_name=self.api_name,
        )

        return result_wav_path

    def _request_arguments(self, text: str, stream: bool) -> dict:
        return {
            "tts_text": text,
            "mode_checkbox_group": self.mode_checkbox_group,
            "sft_dropdown": self.sft_dropdown,
            "prompt_text": self.prompt_text,
            "prompt_wav_upload": self.prompt_wav_upload,
            "prompt_wav_record": self.prompt_wav_record,
            "instruct_text": self.instruct_text,
            "stream": stream,
            "seed": self.seed,
            "speed": self.speed,
            "voice_cache_id": self.voice_cache_id,
            "api_name": self.api_name,
        }

    @staticmethod
    def _audio_path(result) -> str:
        """Normalize Gradio audio outputs across client versions."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("path", "name"):
                if result.get(key):
                    return result[key]
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return TTSEngine._audio_path(result[0])
        raise TypeError(f"Unexpected CosyVoice audio output: {type(result)!r}")

    async def async_stream_audio(self, text: str) -> AsyncIterator[str]:
        """Consume each yield from the Gradio generator without blocking asyncio."""
        if not self.supports_streaming:
            raise RuntimeError("CosyVoice streaming is disabled in configuration")

        job = await asyncio.to_thread(
            self.client.submit,
            **self._request_arguments(text, stream=True),
        )
        output_index = 0
        try:
            while True:
                outputs = job.outputs()
                while output_index < len(outputs):
                    yield self._audio_path(outputs[output_index])
                    output_index += 1

                if job.done():
                    error = job.exception()
                    if error is not None:
                        raise error
                    break
                await asyncio.sleep(0.01)
        finally:
            if not job.done():
                job.cancel()
