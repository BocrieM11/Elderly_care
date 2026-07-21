"""Browser-facing proxy for the portable ASR and TTS services."""

import asyncio
import json

import requests
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/media", tags=["media"])

ASR_URL = "http://127.0.0.1:10095"
TTS_URL = "http://127.0.0.1:50000"
TTS_SAMPLE_RATE = 44100
ASR_WS_URL = "ws://127.0.0.1:10095/v1/vad"


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@router.get("/health")
async def media_health():
    def check(url: str) -> bool:
        try:
            return requests.get(url, timeout=3).ok
        except requests.RequestException:
            return False

    def check_asr() -> bool:
        try:
            response = requests.get(f"{ASR_URL}/health", timeout=3)
            response.raise_for_status()
            status = response.json()
            return (
                status.get("status") == "ok"
                and status.get("vad_ready") is True
                and status.get("kws_ready") is True
            )
        except (requests.RequestException, ValueError):
            return False

    asr_ok, tts_ok = await asyncio.gather(
        asyncio.to_thread(check_asr),
        asyncio.to_thread(check, f"{TTS_URL}/v1/health"),
    )
    return {
        "asr": asr_ok,
        "vad": asr_ok,
        "kws": asr_ok,
        "tts": tts_ok,
        "tts_sample_rate": TTS_SAMPLE_RATE,
    }


@router.post("/transcribe")
async def transcribe(request: Request):
    wav = await request.body()
    if not wav:
        raise HTTPException(status_code=400, detail="Empty audio body")

    def call_asr():
        return requests.post(
            f"{ASR_URL}/transcribe",
            data=wav,
            headers={"Content-Type": "audio/wav"},
            timeout=120,
        )

    try:
        response = await asyncio.to_thread(call_asr)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ASR unavailable: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=502, detail=response.text)
    return response.json()


@router.websocket("/listen")
async def listen(websocket: WebSocket):
    """Bridge browser PCM frames to the local FunASR streaming VAD service."""
    await websocket.accept()
    try:
        async with websockets.connect(
            ASR_WS_URL,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as upstream:
            async def browser_to_asr():
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        await upstream.send(message["text"])

            async def asr_to_browser():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = {
                asyncio.create_task(browser_to_asr()),
                asyncio.create_task(asr_to_browser()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"ASR VAD unavailable: {exc}"})
            )
            await websocket.close(code=1011)
        except Exception:
            pass


@router.post("/tts")
async def synthesize(payload: TTSRequest):
    request_body = {
        "text": payload.text,
        "chunk_length": 100,
        "format": "wav",
        "reference_id": "olv_263_oral5",
        "use_memory_cache": "on",
        "streaming": True,
        "normalize": True,
        "max_new_tokens": 1024,
        "seed": 42,
        "top_p": 0.6,
        "repetition_penalty": 1.2,
        "temperature": 0.25,
    }

    def fetch_audio():
        response = requests.post(
            f"{TTS_URL}/v1/tts",
            json=request_body,
            timeout=(10, 120),
        )
        try:
            if not response.ok:
                return response.status_code, response.text, b""
            return response.status_code, "", response.content
        finally:
            response.close()

    try:
        status_code, detail, audio = await asyncio.to_thread(fetch_audio)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"TTS unavailable: {exc}") from exc
    if not 200 <= status_code < 300:
        raise HTTPException(status_code=502, detail=detail)
    if not audio:
        raise HTTPException(status_code=502, detail="TTS returned empty audio")

    # Fish Speech produces a sentence nearly all at once, so buffering here
    # gives the browser a complete PCM frame with a reliable Content-Length.
    return Response(
        content=audio,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Audio-Format": "pcm_s16le",
            "X-Audio-Sample-Rate": str(TTS_SAMPLE_RATE),
        },
    )
