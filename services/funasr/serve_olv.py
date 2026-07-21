import argparse
import asyncio
import io
import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("funasr-olv")

MODEL = None
VAD_MODEL = None
KWS_MODEL = None
MODEL_LOCK = threading.Lock()
VAD_MODEL_LOCK = threading.Lock()
KWS_MODEL_LOCK = threading.Lock()
SETTINGS = None

SAMPLE_RATE = 16000
VAD_CHUNK_MS = 60
PRE_ROLL_MS = 800
MIN_UTTERANCE_MS = 300
MAX_UTTERANCE_MS = 45000
KWS_PRE_ROLL_MS = 2000
WAKE_ENDPOINT_SILENCE_MS = 650


def transcribe_samples(audio: np.ndarray) -> tuple[str, float]:
    """Run the existing Fun-ASR model on one VAD-confirmed utterance."""
    if MODEL is None:
        raise RuntimeError("ASR model is not ready")

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    started = time.perf_counter()
    with MODEL_LOCK:
        result = MODEL.generate(
            input=[torch.from_numpy(audio)],
            cache={},
            batch_size=1,
            language=SETTINGS.language,
            itn=SETTINGS.use_itn,
            hotwords=SETTINGS.hotwords,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    text = result[0].get("text", "").strip() if result else ""
    return text, elapsed_ms


def load_model() -> None:
    global MODEL, VAD_MODEL, KWS_MODEL

    from funasr import AutoModel

    logger.info("Loading Fun-ASR model from %s", SETTINGS.model)
    logger.info("Device: %s", SETTINGS.device)
    MODEL = AutoModel(
        model=SETTINGS.model,
        trust_remote_code=True,
        remote_code=SETTINGS.remote_code,
        device=SETTINGS.device,
        hub="ms",
        disable_update=True,
    )
    logger.info("Fun-ASR model loaded")

    logger.info("Loading streaming VAD model from %s", SETTINGS.vad_model)
    VAD_MODEL = AutoModel(
        model=SETTINGS.vad_model,
        device=SETTINGS.vad_device,
        hub="ms",
        disable_update=True,
        disable_pbar=True,
    )
    logger.info("Streaming FSMN-VAD loaded on %s", SETTINGS.vad_device)

    import sherpa_onnx

    kws_dir = Path(SETTINGS.kws_model_dir)
    logger.info("Loading sherpa-onnx streaming KWS from %s", kws_dir)
    KWS_MODEL = sherpa_onnx.KeywordSpotter(
        tokens=str(kws_dir / "tokens.txt"),
        encoder=str(kws_dir / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(kws_dir / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(kws_dir / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        keywords_file=SETTINGS.kws_keywords_file,
        num_threads=SETTINGS.kws_num_threads,
        max_active_paths=SETTINGS.kws_max_active_paths,
        keywords_score=SETTINGS.kws_score,
        keywords_threshold=SETTINGS.kws_threshold,
        num_trailing_blanks=SETTINGS.kws_trailing_blanks,
        provider="cpu",
    )
    logger.info("Streaming KWS loaded on CPU")

    if SETTINGS.warmup:
        warmup_audio = Path(SETTINGS.model) / "example" / "zh.mp3"
        if warmup_audio.is_file():
            started = time.perf_counter()
            with MODEL_LOCK:
                MODEL.generate(
                    input=[str(warmup_audio)],
                    cache={},
                    batch_size=1,
                    language=SETTINGS.language,
                    itn=SETTINGS.use_itn,
                )
            logger.info("Warmup completed in %.3fs", time.perf_counter() - started)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Fun-ASR for Open-LLM-VTuber", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None and VAD_MODEL is not None and KWS_MODEL is not None else "loading",
        "model": SETTINGS.model,
        "vad_model": SETTINGS.vad_model,
        "vad_device": SETTINGS.vad_device,
        "vad_ready": VAD_MODEL is not None,
        "kws_ready": KWS_MODEL is not None,
        "kws_model": SETTINGS.kws_model_dir,
        "kws_keywords_file": SETTINGS.kws_keywords_file,
        "kws_chunk_latency_ms": 160,
        "kws_score": SETTINGS.kws_score,
        "kws_threshold": SETTINGS.kws_threshold,
        "kws_trailing_blanks": SETTINGS.kws_trailing_blanks,
        "device": SETTINGS.device,
        "hotwords": SETTINGS.hotwords,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")

    try:
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV audio: {exc}") from exc

    if sample_rate != 16000:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 16000 Hz audio, received {sample_rate} Hz",
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    try:
        text, elapsed_ms = await asyncio.to_thread(transcribe_samples, audio)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Do not persist recognized room speech in service logs. Conversation text
    # is stored only after the browser has deliberately entered an active turn.
    logger.info("Transcribed %.2fs audio in %.1fms", len(audio) / sample_rate, elapsed_ms)
    return {"text": text, "elapsed_ms": elapsed_ms}


class StreamingVadSession:
    """Per-browser FSMN-VAD state and bounded utterance audio buffer."""

    def __init__(self, wake_transition: bool = False):
        from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

        # Elderly speakers often pause inside a sentence. Short utterances get a
        # generous endpoint delay while longer turns are cut more aggressively.
        silence_schedule = [
            (5000, WAKE_ENDPOINT_SILENCE_MS if wake_transition else 1400),
            (15000, 1100),
            (30000, 800),
            (float("inf"), 500),
        ]
        self.vad = DynamicStreamingVAD(
            VAD_MODEL,
            chunk_size_ms=VAD_CHUNK_MS,
            silence_schedule=silence_schedule,
            sample_rate=SAMPLE_RATE,
        )
        self.pre_roll = deque()
        self.pre_roll_samples = 0
        self.total_samples = 0
        self.utterance_chunks = []
        self.utterance_buffer_start = 0

    def reset(self) -> None:
        self.vad.reset()
        self.pre_roll.clear()
        self.pre_roll_samples = 0
        self.total_samples = 0
        self.utterance_chunks = []
        self.utterance_buffer_start = 0

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        self.pre_roll.append(samples)
        self.pre_roll_samples += len(samples)
        limit = int(SAMPLE_RATE * PRE_ROLL_MS / 1000)
        while self.pre_roll and self.pre_roll_samples - len(self.pre_roll[0]) >= limit:
            self.pre_roll_samples -= len(self.pre_roll.popleft())

    def feed(self, samples: np.ndarray) -> tuple[bool, list[list[int]], np.ndarray | None]:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(samples):
            return False, [], None

        was_speaking = self.vad.is_speaking
        self._remember_pre_roll(samples)
        self.total_samples += len(samples)

        with VAD_MODEL_LOCK:
            segments = self.vad.feed(torch.from_numpy(samples))

        started = not was_speaking and self.vad.is_speaking
        if started:
            self.utterance_chunks = list(self.pre_roll)
            self.utterance_buffer_start = self.total_samples - self.pre_roll_samples
        elif was_speaking:
            self.utterance_chunks.append(samples)

        # Some very short segments can start and end inside one feed call.
        if segments and not self.utterance_chunks:
            self.utterance_chunks = list(self.pre_roll)
            self.utterance_buffer_start = self.total_samples - self.pre_roll_samples

        if not segments:
            if self.utterance_chunks:
                max_samples = int(SAMPLE_RATE * MAX_UTTERANCE_MS / 1000)
                current_samples = sum(len(chunk) for chunk in self.utterance_chunks)
                if current_samples >= max_samples:
                    audio = np.concatenate(self.utterance_chunks)[:max_samples]
                    self.reset()
                    return started, [[0, MAX_UTTERANCE_MS]], audio
            return started, [], None

        start_ms, end_ms = segments[-1]
        buffered = np.concatenate(self.utterance_chunks)
        start_sample = max(0, int(start_ms * SAMPLE_RATE / 1000) - self.utterance_buffer_start)
        end_sample = min(len(buffered), int(end_ms * SAMPLE_RATE / 1000) - self.utterance_buffer_start)
        audio = buffered[start_sample:end_sample]
        self.reset()
        return started, segments, audio


class StreamingKwsSession:
    """Per-browser low-cost KWS stream plus audio needed for wake hand-off."""

    def __init__(self):
        self.pre_roll = deque()
        self.pre_roll_samples = 0
        with KWS_MODEL_LOCK:
            self.stream = KWS_MODEL.create_stream()

    def reset(self) -> None:
        self.pre_roll.clear()
        self.pre_roll_samples = 0
        with KWS_MODEL_LOCK:
            self.stream = KWS_MODEL.create_stream()

    def _remember_pre_roll(self, samples: np.ndarray) -> None:
        self.pre_roll.append(samples)
        self.pre_roll_samples += len(samples)
        limit = int(SAMPLE_RATE * KWS_PRE_ROLL_MS / 1000)
        while self.pre_roll and self.pre_roll_samples - len(self.pre_roll[0]) >= limit:
            self.pre_roll_samples -= len(self.pre_roll.popleft())

    def take_pre_roll(self) -> np.ndarray:
        if not self.pre_roll:
            return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(self.pre_roll), dtype=np.float32)

    def feed(self, samples: np.ndarray) -> str:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if not len(samples):
            return ""
        self._remember_pre_roll(samples)
        with KWS_MODEL_LOCK:
            self.stream.accept_waveform(SAMPLE_RATE, samples)
            while KWS_MODEL.is_ready(self.stream):
                KWS_MODEL.decode_stream(self.stream)
                result = KWS_MODEL.get_result(self.stream)
                if result:
                    KWS_MODEL.reset_stream(self.stream)
                    return str(result)
        return ""


@app.websocket("/v1/vad")
async def stream_vad(websocket: WebSocket):
    """Run CPU KWS while asleep and warm FunASR/VAD only after activation."""
    await websocket.accept()
    if MODEL is None or VAD_MODEL is None or KWS_MODEL is None:
        await websocket.send_json(
            {"type": "error", "message": "ASR, VAD, or KWS model is not ready"}
        )
        await websocket.close(code=1013)
        return

    mode = "standby"
    kws_session = StreamingKwsSession()
    vad_session = None
    wake_transition = False
    await websocket.send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "chunk_ms": VAD_CHUNK_MS,
            "endpoint": "sherpa-kws+funasr-fsmn-vad",
            "mode": mode,
            "kws_latency_ms": 160,
            "wake_words": SETTINGS.hotwords,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "reset":
                    if mode == "standby":
                        kws_session.reset()
                    elif vad_session is not None:
                        vad_session.reset()
                    await websocket.send_json({"type": "listening", "mode": mode})
                elif control.get("type") == "sleep":
                    mode = "standby"
                    kws_session = StreamingKwsSession()
                    vad_session = None
                    wake_transition = False
                    await websocket.send_json({"type": "listening", "mode": mode})
                elif control.get("type") == "activate":
                    mode = "conversation"
                    vad_session = StreamingVadSession()
                    wake_transition = False
                    await websocket.send_json({"type": "listening", "mode": mode})
                elif control.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            pcm = message.get("bytes")
            if not pcm or len(pcm) % 2:
                continue
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

            if mode == "standby":
                keyword = await asyncio.to_thread(kws_session.feed, samples)
                if not keyword:
                    continue
                pre_roll = kws_session.take_pre_roll()
                mode = "conversation"
                wake_transition = True
                vad_session = StreamingVadSession(wake_transition=True)
                await websocket.send_json(
                    {
                        "type": "wake_word",
                        "phrase": keyword,
                        "mode": mode,
                        "detected_at_ms": round(time.time() * 1000),
                    }
                )
                started, segments, utterance = await asyncio.to_thread(
                    vad_session.feed, pre_roll
                )
            else:
                started, segments, utterance = await asyncio.to_thread(
                    vad_session.feed, samples
                )

            if started:
                await websocket.send_json({"type": "speech_start"})
            if utterance is None:
                continue

            duration_ms = round(len(utterance) * 1000 / SAMPLE_RATE)
            await websocket.send_json(
                {"type": "speech_end", "duration_ms": duration_ms, "segments": segments}
            )
            if duration_ms < MIN_UTTERANCE_MS:
                await websocket.send_json({"type": "no_speech", "duration_ms": duration_ms})
                if wake_transition:
                    vad_session = StreamingVadSession()
                    wake_transition = False
                continue

            try:
                text, elapsed_ms = await asyncio.to_thread(transcribe_samples, utterance)
                logger.info(
                    "VAD utterance %.2fs transcribed in %.1fms",
                    duration_ms / 1000,
                    elapsed_ms,
                )
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "duration_ms": duration_ms,
                        "elapsed_ms": elapsed_ms,
                        "wake_transition": wake_transition,
                    }
                )
                if wake_transition:
                    vad_session = StreamingVadSession()
                    wake_transition = False
            except Exception as exc:
                logger.exception("Streaming transcription failed")
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("VAD WebSocket failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Fun-ASR service for Open-LLM-VTuber")
    parser.add_argument(
        "--model",
        default=r"../../models/Fun-ASR-Nano-2512",
    )
    parser.add_argument("--remote-code", default="./model.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10095)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--vad-device", default="cpu")
    parser.add_argument(
        "--kws-model-dir",
        default=r"../../models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
    )
    parser.add_argument("--kws-keywords-file", default="./keywords.txt")
    parser.add_argument("--kws-num-threads", type=int, default=1)
    parser.add_argument("--kws-max-active-paths", type=int, default=4)
    parser.add_argument("--kws-score", type=float, default=1.5)
    parser.add_argument("--kws-threshold", type=float, default=0.25)
    parser.add_argument("--kws-trailing-blanks", type=int, default=1)
    parser.add_argument("--language", default="中文")
    # Keep Chinese defaults in the UTF-8 Python source. Windows PowerShell 5 can
    # corrupt non-ASCII native-process arguments depending on its active codepage.
    parser.add_argument(
        "--hotwords",
        nargs="*",
        default=["你好暖暖", "暖暖你好", "小暖同学"],
    )
    parser.add_argument("--no-itn", dest="use_itn", action="store_false")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(use_itn=True, warmup=True)
    return parser.parse_args()


if __name__ == "__main__":
    SETTINGS = parse_args()
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, workers=1)
