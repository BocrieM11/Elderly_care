import os
import json
import re
import secrets
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Literal
from uuid import uuid4
import numpy as np
from fastapi import (
    APIRouter,
    WebSocket,
    UploadFile,
    File,
    Response,
    Header,
    HTTPException,
    Request,
)
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler


class DetectionEvent(BaseModel):
    """External events accepted from the video detection system."""

    event_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    event_type: Literal["sit_up", "fall_detected"]
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value


def init_client_ws_route(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)
    accepted_detection_events: OrderedDict[str, float] = OrderedDict()
    detection_consumers: dict[str, WebSocket] = {}
    project_root = Path(__file__).resolve().parents[3]
    configured_log_path = os.getenv("OLV_DETECTION_EVENT_LOG")
    detection_event_log = (
        Path(configured_log_path).expanduser()
        if configured_log_path
        else project_root / "logs" / "detection_events.jsonl"
    )
    if not detection_event_log.is_absolute():
        detection_event_log = project_root / detection_event_log
    detection_event_log.parent.mkdir(parents=True, exist_ok=True)
    detection_event_log.touch(exist_ok=True)
    detection_event_log_lock = Lock()

    def append_detection_event_log(
        request: Request,
        raw_event: dict,
        status: str,
        http_status: int,
        detail: str,
        delivery_target: str | None,
    ) -> None:
        """Persist one valid event request and its delivery result as JSONL."""
        record = {
            "received_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_ip": request.client.host if request.client else None,
            "event": raw_event,
            "delivery_status": status,
            "delivery_target": delivery_target,
            "http_status": http_status,
            "detail": detail,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with detection_event_log_lock:
                with detection_event_log.open("a", encoding="utf-8") as log_file:
                    log_file.write(line + "\n")
        except Exception as error:
            logger.error(f"Failed to persist detection event log: {error}")

    def remember_detection_event(event_id: str) -> None:
        now = monotonic()
        cutoff = now - 24 * 60 * 60
        while accepted_detection_events:
            _, accepted_at = next(iter(accepted_detection_events.items()))
            if accepted_at >= cutoff:
                break
            accepted_detection_events.popitem(last=False)
        accepted_detection_events[event_id] = now
        while len(accepted_detection_events) > 2048:
            accepted_detection_events.popitem(last=False)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    @router.get("/api/detection-events/health")
    async def detection_events_health():
        """Connectivity check for the remote video detection computer."""
        return {
            "status": "ok",
            "accepted_event_types": ["sit_up", "fall_detected"],
            "connected_olv_clients": (
                len(ws_handler.client_connections) + len(detection_consumers)
            ),
        }

    @router.websocket("/api/detection-events/ws")
    async def detection_events_websocket(websocket: WebSocket, consumer_id: str):
        """Keep the active companion page connected for event delivery."""
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", consumer_id):
            await websocket.close(code=1008)
            return

        await websocket.accept()
        detection_consumers[consumer_id] = websocket
        await websocket.send_json({"type": "ready"})
        try:
            while True:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            if detection_consumers.get(consumer_id) is websocket:
                detection_consumers.pop(consumer_id, None)

    @router.post("/api/detection-events")
    async def receive_detection_event(
        request: Request,
        event: DetectionEvent,
        x_olv_event_token: str | None = Header(default=None),
    ):
        """Accept a sit-up or fall event and trigger proactive speech."""
        configured_token = os.getenv("OLV_DETECTION_API_TOKEN")
        if configured_token and not secrets.compare_digest(
            x_olv_event_token or "", configured_token
        ):
            raise HTTPException(status_code=401, detail="invalid event token")

        raw_event = await request.json()
        event_data = event.model_dump(mode="json")

        def response(
            status: str,
            detail: str,
            status_code: int,
            delivery_target: str | None = None,
        ) -> JSONResponse:
            append_detection_event_log(
                request=request,
                raw_event=raw_event,
                status=status,
                http_status=status_code,
                detail=detail,
                delivery_target=delivery_target,
            )
            return JSONResponse(
                {
                    "status": status,
                    "event_id": event.event_id,
                    "detail": detail,
                },
                status_code=status_code,
            )

        if event.event_id in accepted_detection_events:
            return response(
                status="duplicate",
                detail="event_id has already been accepted",
                status_code=200,
            )

        for consumer_id, consumer_ws in reversed(list(detection_consumers.items())):
            try:
                await consumer_ws.send_json(
                    {"type": "detection-event", "event": event_data}
                )
                remember_detection_event(event.event_id)
                return response(
                    status="triggered",
                    detail="event delivered to the active companion page",
                    status_code=202,
                    delivery_target="companion_page",
                )
            except Exception as error:
                logger.warning(
                    f"Failed to deliver detection event to {consumer_id}: {error}"
                )
                if detection_consumers.get(consumer_id) is consumer_ws:
                    detection_consumers.pop(consumer_id, None)

        result = await ws_handler.trigger_external_event(event_data)
        if result["status"] == "no_client_connected":
            return response(
                status=result["status"],
                detail=result["detail"],
                status_code=503,
            )
        remember_detection_event(event.event_id)
        if result["status"] == "duplicate":
            return response(
                status=result["status"],
                detail=result["detail"],
                status_code=200,
            )
        return response(
            status=result["status"],
            detail=result["detail"],
            status_code=202,
            delivery_target="olv_websocket",
        )

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    return router
