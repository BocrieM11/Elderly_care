import asyncio
import json
from typing import Any, Awaitable, Dict, List, Optional, Callable

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from prompts import prompt_loader


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
    request_camera_snapshot: Optional[
        Callable[[], Awaitable[Optional[List[Dict[str, Any]]]]]
    ] = None,
) -> None:
    """Handle triggers that start a conversation"""
    metadata = None

    if msg_type == "ai-speak-signal":
        external_event = data.get("external_event")
        event_type = (
            external_event.get("event_type")
            if isinstance(external_event, dict)
            else None
        )
        if event_type == "sit_up":
            occurred_at = external_event.get("occurred_at", "未知时间")
            user_input = (
                "这是一个内部环境事件，不是用户说的话：系统确认用户刚刚从床上坐起。"
                f"事件发生时间是 {occurred_at}。请自然、温和、简短地主动开启对话，"
                "用一到两句话打招呼并轻轻询问睡眠或身体感受。"
                "不要提到摄像头、检测系统、事件、报警、一级离床或坐起识别，"
                "不要断言用户生病或处于危险中，也不要一次连续追问多个问题。"
            )
        elif event_type == "fall_detected":
            occurred_at = external_event.get("occurred_at", "未知时间")
            user_input = (
                "这是一个内部安全事件，不是用户说的话：系统发现用户可能刚刚跌倒。"
                f"事件发生时间是 {occurred_at}。请立即用镇定、清晰、简短的语气提醒用户"
                "先不要急着起身，并询问是否撞到头、明显疼痛、出血或无法活动。"
                "不要提到摄像头、检测系统、算法或事件，不要责备或制造恐慌，"
                "你没有拨打电话、呼叫急救或联系任何人的能力。绝对不要提出替用户叫"
                "救护车、报警、联系家属或护工，也不要询问是否要你代为联系。"
                "如果用户可能严重受伤或不能安全起身，只能明确指导用户自己拨打120，"
                "或呼喊身边的人、家属或护工前来帮助。"
            )
        else:
            try:
                # Get proactive speak prompt from config
                prompt_name = "proactive_speak_prompt"
                prompt_file = context.system_config.tool_prompts.get(prompt_name)
                if prompt_file:
                    user_input = prompt_loader.load_util(prompt_file)
                else:
                    logger.warning(
                        "Proactive speak prompt not configured, using default"
                    )
                    user_input = "Please say something."
            except Exception as e:
                logger.error(f"Error loading proactive speak prompt: {e}")
                user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }

        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    session_emoji=session_emoji,
                    metadata=metadata,
                    request_camera_snapshot=request_camera_snapshot,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
                request_camera_snapshot=request_camera_snapshot,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
