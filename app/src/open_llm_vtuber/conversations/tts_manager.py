import asyncio
import json
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._pending_short_speech = None

    @staticmethod
    def _speakable_char_count(text: str) -> int:
        return len(re.sub(r"[^\w\u4e00-\u9fff]+", "", text))

    @staticmethod
    def _merge_actions(first: Optional[Actions], second: Optional[Actions]):
        if first is None:
            return second
        if second is None:
            return first
        return Actions(
            expressions=(first.expressions or []) + (second.expressions or []),
            pictures=(first.pictures or []) + (second.pictures or []),
            sounds=(first.sounds or []) + (second.sounds or []),
        )

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        force: bool = False,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if not force and self._pending_short_speech is not None:
            pending = self._pending_short_speech
            self._pending_short_speech = None
            tts_text = pending["tts_text"] + tts_text
            display_text = DisplayText(
                text=pending["display_text"].text + display_text.text,
                name=pending["display_text"].name,
                avatar=pending["display_text"].avatar,
            )
            actions = self._merge_actions(pending["actions"], actions)

        if (
            not force
            # Streaming engines can deliver a short opening phrase quickly.
            # Holding it to merge with the next sentence defeats first-audio latency.
            and not getattr(tts_engine, "supports_streaming", False)
            and self._speakable_char_count(tts_text) < 4
            and self._speakable_char_count(tts_text) > 0
        ):
            self._pending_short_speech = {
                "tts_text": tts_text,
                "display_text": display_text,
                "actions": actions,
                "live2d_model": live2d_model,
                "tts_engine": tts_engine,
                "websocket_send": websocket_send,
            }
            logger.debug(f"Buffering short TTS fragment: '''{tts_text}'''")
            return

        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # Create and queue the TTS task
        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=current_sequence,
            )
        )
        self.task_list.append(task)

    async def flush_pending(self) -> None:
        """Queue a buffered short fragment when no following sentence arrives."""
        if self._pending_short_speech is None:
            return
        pending = self._pending_short_speech
        self._pending_short_speech = None
        await self.speak(**pending, force=True)

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Runs continuously until all payloads are processed.
        """
        buffered_payloads: Dict[
            int, Deque[Tuple[Dict, bool, Optional[asyncio.Future]]]
        ] = defaultdict(deque)

        while True:
            try:
                # Get payload from queue
                payload, sequence_number, is_final, acknowledgement = (
                    await self._payload_queue.get()
                )
                buffered_payloads[sequence_number].append(
                    (payload, is_final, acknowledgement)
                )

                # Send payloads in order
                while buffered_payloads.get(self._next_sequence_to_send):
                    next_payload, next_is_final, next_acknowledgement = (
                        buffered_payloads[self._next_sequence_to_send].popleft()
                    )
                    await websocket_send(json.dumps(next_payload))
                    if next_acknowledgement and not next_acknowledgement.done():
                        next_acknowledgement.set_result(None)
                    if next_is_final:
                        buffered_payloads.pop(self._next_sequence_to_send, None)
                        self._next_sequence_to_send += 1

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number, True, None))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        if tts_engine.supports_streaming:
            await self._process_streaming_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                tts_engine=tts_engine,
                sequence_number=sequence_number,
            )
            return

        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
                leading_silence_ms=80 if sequence_number == 0 else 0,
            )
            # Queue the payload with its sequence number
            acknowledgement = asyncio.get_running_loop().create_future()
            await self._payload_queue.put(
                (payload, sequence_number, True, acknowledgement)
            )
            await acknowledgement

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            acknowledgement = asyncio.get_running_loop().create_future()
            await self._payload_queue.put(
                (payload, sequence_number, True, acknowledgement)
            )
            await acknowledgement

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _process_streaming_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Forward each generated chunk and finish the sequence with an end marker."""
        stream_id = f"{sequence_number}-{uuid.uuid4().hex[:12]}"
        chunk_index = 0
        first_chunk = True
        stream_error = None

        logger.debug(f"Streaming audio for '''{tts_text}''' as {stream_id}")
        try:
            async for audio_file_path in tts_engine.async_stream_audio(tts_text):
                try:
                    payload = prepare_audio_payload(
                        audio_path=audio_file_path,
                        display_text=display_text if first_chunk else None,
                        actions=actions if first_chunk else None,
                        leading_silence_ms=(
                            80 if sequence_number == 0 and first_chunk else 0
                        ),
                    )
                    payload.update(
                        {
                            "type": "audio-chunk",
                            "stream_id": stream_id,
                            "chunk_index": chunk_index,
                            "is_first": first_chunk,
                            "is_final": False,
                        }
                    )
                    await self._payload_queue.put(
                        (payload, sequence_number, False, None)
                    )
                    first_chunk = False
                    chunk_index += 1
                finally:
                    tts_engine.remove_file(audio_file_path, verbose=False)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            stream_error = error
            logger.exception(f"CosyVoice streaming failed for {stream_id}: {error}")

        # A separate marker lets the first audio chunk be sent immediately; we do
        # not have to hold it back just to discover whether it is the final chunk.
        end_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text if first_chunk else None,
            actions=actions if first_chunk else None,
        )
        end_payload.update(
            {
                "type": "audio-chunk",
                "stream_id": stream_id,
                "chunk_index": chunk_index,
                "is_first": first_chunk,
                "is_final": True,
                "error": str(stream_error) if stream_error else None,
            }
        )
        acknowledgement = asyncio.get_running_loop().create_future()
        await self._payload_queue.put(
            (end_payload, sequence_number, True, acknowledgement)
        )
        await acknowledgement

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text"""
        logger.debug(f"🏃Generating audio for '''{text}'''...")
        return await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        for task in self.task_list:
            if not task.done():
                task.cancel()
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._pending_short_speech = None
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
