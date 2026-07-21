import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from loguru import logger

from .input_types import ImageData, ImageSource
from .stateless_llm.stateless_llm_interface import StatelessLLMInterface

# Frames below this score are treated as camera noise / an unchanged scene.
# Do not skip the first frame: it is needed to establish the initial state.
_SCENE_UPDATE_MOTION_THRESHOLD = 0.05


class VisualQueryType(str, Enum):
    NONE = "none"
    CURRENT = "current"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class VisualObservation:
    query_type: VisualQueryType
    answerable: bool
    direct_answer: str
    observations: List[str]
    confidence: float
    answerability: str = "full"

    @classmethod
    def unavailable(
        cls, query_type: VisualQueryType, reason: str
    ) -> "VisualObservation":
        return cls(
            query_type=query_type,
            answerable=False,
            direct_answer="我现在没看清。",
            observations=[reason],
            confidence=0.0,
            answerability="none",
        )

    def to_chat_context(self) -> str:
        evidence = "；".join(self.observations) if self.observations else "无"
        return (
            "以下是内部视觉观察器针对本轮问题提供的事实证据。"
            "不要向用户解释观察器、图像、帧或处理过程。\n"
            f"问题类型：{self.query_type.value}\n"
            f"回答完整度：{self.answerability}\n"
            f"直接结论：{self.direct_answer}\n"
            f"已确认的可见事实：{evidence}\n"
            f"置信度：{self.confidence:.2f}\n"
            "回答规则：full 时直接回答；partial 时必须先肯定地列出已确认事实，"
            "再说明其余细节可能没有完整捕捉，绝不能用‘没看清’否定已确认事实；"
            "只有 none 且没有可靠事实时才能说没有捕捉清楚。"
            "不得用无关的情绪猜测代替答案。"
        )


@dataclass(frozen=True)
class VisualSceneState:
    captured_at_ms: int
    subject_visible: bool
    scene_summary: str
    appearance: List[str]
    objects: List[str]
    current_action: str
    expression: str
    confidence: float


@dataclass(frozen=True)
class VisualEvent:
    captured_at_ms: int
    description: str
    event_type: str = "scene_change"
    before: str = ""
    after: str = ""
    confidence: float = 0.0


class VisualObserver:
    """Maintain a compact visual state and answer turn-specific visual questions."""

    _QUESTION_SYSTEM_PROMPT = """你是视觉观察器，不是聊天助手。
你的唯一任务是根据提供的视觉内容和用户问题，提取可见事实并给出直接结论。

规则：
1. 只能依据确实可见的内容，不补充背景故事，不寒暄，不反问。
2. 不要猜测用户是否困惑、紧张、开心，或正在思考什么；除非用户明确询问可见表情。
3. current 表示判断当前状态，应主要依据最接近当前的内容。
4. temporal 表示判断一段连续过程，应比较时间顺序并描述可见变化。
5. 用户询问看起来偏男性还是女性时，可以根据当前外观给出“外观看起来更偏男性/女性/无法判断”，但不要把外观判断说成真实身份事实。
6. answerability 只能是 full、partial、none：证据足够时为 full；有可靠事实但不完整时为 partial；完全没有可靠事实时才为 none。
7. partial 时 direct_answer 必须先说出已确认事实，不能只写“没看清”。
8. 只输出一个 JSON 对象，不要使用 Markdown，不要输出任何额外文字。

JSON 格式：
{
  "answerability": "full",
  "direct_answer": "针对用户问题的一句直接结论",
  "observations": ["可见事实1", "可见事实2"],
  "confidence": 0.0
}
"""

    _SCENE_SYSTEM_PROMPT = """你是视频通话中的后台视觉感知模块，不是聊天助手。
请把当前摄像头画面压缩成可供对话模型使用的稳定场景状态，并判断相对上一状态是否出现了值得记住的可见变化。

规则：
1. 只记录画面中确实可见的事实，不推测身份、职业、心情、意图或正在思考的事情。
2. appearance 记录较稳定的外观事实，如发型、眼镜、衣着和画面中呈现的外观特征。
3. current_action 只写当前正在做的可见动作；expression 只写能直接观察到的嘴角、眼睛等表情特征，不得写开心、平静、困惑等心理或情绪结论。
4. changed 仅在人物出现/离开、明显动作、手势、拿起或放下物体、衣着或场景显著变化时为 true。轻微光照和镜头噪声不算变化。
5. event 只在 changed 为 true 时写一句简短事件，否则为空字符串。
6. 看不清的字段留空，不要编造。
7. 只输出一个 JSON 对象，不要使用 Markdown或额外文字。

JSON 格式：
{
  "subject_visible": true,
  "scene_summary": "一句当前场景概述",
  "appearance": ["稳定可见事实1"],
  "objects": ["可见物体1"],
  "current_action": "当前动作",
  "expression": "可见表情",
  "confidence": 0.0,
  "changed": false,
  "event": ""
}
"""

    _TURN_SYSTEM_PROMPT = """你是视频通话中的实时视觉感知模块，不是聊天助手。
你会收到：上一份结构化场景状态与事件、用户本轮问题、以及本轮 ASR 完成后采集的一张当前摄像头画面。
请用这唯一一张当前画面同时完成两件事：刷新场景状态；为本轮问题提取必要的视觉事实。

规则：
1. 只能依据当前画面和给出的历史结构化事件，不得假装看过未记录的过去画面。
2. 不推测身份、职业、心情、意图或正在思考的事情；表情只能描述嘴角、眼睛等可见特征。
3. scene_change 仅记录人物出现/离开、明显动作或手势、拿起/放下或更换物体、衣着或场景显著变化。
4. 用户问过去发生了什么时，只能使用给出的事件记录；有部分可靠事件就设 answerability=partial 并先列出它们，不能因为记录可能不完整就说什么都没看清。
5. 用户问外观看起来偏男性还是女性时，可以给出外观倾向，但不能说成真实身份事实。
6. 问题与现场画面无关时 visual_relevance=false，不要硬把画面内容带进回答。
7. answerability 只能是 full、partial、none。用户使用“都、全部、具体”等词但事件记录不完整时应选 partial；只有完全没有可靠事实时才选 none。
8. partial 的 direct_answer 必须先肯定地列出已确认动作，再说明可能不完整，不得以“没看清”开头。
9. 只输出一个 JSON 对象，不要使用 Markdown，不要输出额外文字。

JSON 格式：
{
  "scene_state": {
    "subject_visible": true,
    "scene_summary": "一句当前场景概述",
    "appearance": ["稳定可见事实"],
    "objects": ["当前可见物体"],
    "current_action": "当前动作",
    "expression": "可见表情特征",
    "confidence": 0.0
  },
  "scene_change": {
    "changed": false,
    "event_type": "object_change",
    "description": "一句变化描述",
    "before": "变化前",
    "after": "变化后",
    "confidence": 0.0
  },
  "question_observation": {
    "visual_relevance": true,
    "answerability": "partial",
    "direct_answer": "针对问题的直接视觉结论",
    "observations": ["可见证据"],
    "confidence": 0.0
  }
}
"""

    _TEMPORAL_HINTS = (
        "刚才",
        "刚刚",
        "之前",
        "过程",
        "变化",
        "做了什么",
        "发生了什么",
        "比划",
        "手势",
        "挥手",
        "动作",
        "拿起",
        "放下",
        "回放",
        "what did i do",
        "what happened",
        "gesture",
        "before",
    )
    _CURRENT_HINTS = (
        "我是男",
        "我是女",
        "男是女",
        "性别",
        "长相",
        "外貌",
        "好看",
        "看起来",
        "你看我",
        "看到我",
        "看见我",
        "镜头里",
        "画面",
        "照片",
        "图片",
        "视频",
        "摄像头",
        "我的表情",
        "我的衣服",
        "我穿",
        "我手里",
        "我拿着",
        "我的发型",
        "我身后",
        "几个人",
        "什么颜色",
        "what do i look like",
        "am i wearing",
        "do you see me",
        "what am i holding",
    )

    def __init__(
        self,
        llm: StatelessLLMInterface,
        current_frame_limit: int = 2,
        temporal_frame_limit: int = 6,
        state_max_age_ms: int = 15_000,
        event_limit: int = 20,
    ) -> None:
        self._llm = llm
        self._current_frame_limit = max(1, current_frame_limit)
        self._temporal_frame_limit = max(2, temporal_frame_limit)
        self._state_max_age_ms = max(1_000, state_max_age_ms)
        self._scene_state: Optional[VisualSceneState] = None
        self._events: Deque[VisualEvent] = deque(maxlen=max(1, event_limit))
        self._llm_lock = asyncio.Lock()

    def clone_empty(self) -> "VisualObserver":
        return VisualObserver(
            llm=self._llm,
            current_frame_limit=self._current_frame_limit,
            temporal_frame_limit=self._temporal_frame_limit,
            state_max_age_ms=self._state_max_age_ms,
            event_limit=self._events.maxlen or 20,
        )

    def clear_state(self) -> None:
        self._scene_state = None
        self._events.clear()

    def has_fresh_state(self) -> bool:
        if self._scene_state is None:
            return False
        return self._state_age_ms() <= self._state_max_age_ms

    def _state_age_ms(self) -> int:
        if self._scene_state is None:
            return 0
        return max(0, int(time.time() * 1000) - self._scene_state.captured_at_ms)

    def get_chat_context(self) -> Optional[str]:
        state = self._scene_state
        if state is None or not self.has_fresh_state():
            return None

        appearance = "；".join(state.appearance) if state.appearance else "未看清"
        objects = "；".join(state.objects) if state.objects else "无明确物体"
        recent_events = [
            event
            for event in self._events
            if state.captured_at_ms - event.captured_at_ms <= 60_000
        ]
        event_text = "；".join(
            f"{event.description}"
            + (f"（{event.before}→{event.after}）" if event.before or event.after else "")
            for event in recent_events
        ) or "无"
        return (
            "【持续视频感知状态（内部上下文）】\n"
            "这是摄像头视频流形成的实时状态，不是用户发送的图片附件。"
            "回答时要像正在视频通话一样自然，绝不能提到图片、帧、截图、视觉输入或处理过程。\n"
            "摄像头中的人物默认是正在与你通话的用户，不是你自己；描述对方时使用“你”，"
            "不得把对方的场景或动作说成“我正在……”。\n"
            f"状态距现在：{self._state_age_ms() / 1000:.1f} 秒\n"
            f"人物是否可见：{'是' if state.subject_visible else '否'}\n"
            f"当前场景：{state.scene_summary or '未看清'}\n"
            f"稳定外观：{appearance}\n"
            f"可见物体：{objects}\n"
            f"当前动作：{state.current_action or '无明确动作'}\n"
            f"可见表情：{state.expression or '未看清'}\n"
            f"最近可见事件：{event_text}\n"
            f"视觉置信度：{state.confidence:.2f}\n"
            "仅在用户问题与现场画面有关时使用这些事实；只描述可见表情特征，"
            "不得由表情推断用户的心情、心理状态或意图。"
        )

    @classmethod
    def classify_query(
        cls, text_prompt: str, images: List[ImageData]
    ) -> VisualQueryType:
        normalized = text_prompt.lower().strip()
        if any(hint in normalized for hint in cls._TEMPORAL_HINTS):
            return VisualQueryType.TEMPORAL
        if any(hint in normalized for hint in cls._CURRENT_HINTS):
            return VisualQueryType.CURRENT
        return VisualQueryType.NONE

    @staticmethod
    def _ordered_camera_images(images: List[ImageData]) -> List[ImageData]:
        return sorted(
            (image for image in images if image.source == ImageSource.CAMERA),
            key=lambda image: (
                image.sequence_index
                if image.sequence_index is not None
                else image.captured_at_ms or 0
            ),
        )

    @staticmethod
    def _select_temporal_images(
        ordered_images: List[ImageData], max_frames: int
    ) -> List[ImageData]:
        if len(ordered_images) <= max_frames:
            return ordered_images

        last_index = len(ordered_images) - 1
        selected_indexes = {0, last_index}
        motion_candidates = sorted(
            range(1, last_index),
            key=lambda index: ordered_images[index].motion_score or 0.0,
            reverse=True,
        )
        for index in motion_candidates:
            if len(selected_indexes) >= max_frames:
                break
            selected_indexes.add(index)

        for slot in range(max_frames):
            if len(selected_indexes) >= max_frames:
                break
            selected_indexes.add(round(slot * last_index / max(1, max_frames - 1)))

        return [ordered_images[index] for index in sorted(selected_indexes)]

    def select_images(
        self, images: List[ImageData], query_type: VisualQueryType
    ) -> List[ImageData]:
        camera_images = self._ordered_camera_images(images)
        other_images = [image for image in images if image.source != ImageSource.CAMERA]
        selected_camera = camera_images[-1:] if query_type != VisualQueryType.NONE else []
        return [*selected_camera, *other_images]

    @staticmethod
    def _time_label(image: ImageData) -> str:
        if image.source == ImageSource.CAMERA:
            relative_seconds = abs(image.relative_time_ms or 0) / 1000
            if relative_seconds < 0.05:
                return "时间：当前。"
            return f"时间：距现在 {relative_seconds:.1f} 秒。"
        if image.source == ImageSource.SCREEN:
            return "来源：当前屏幕。"
        return "来源：用户提供的当前视觉内容。"

    @staticmethod
    def _image_block(image: ImageData) -> Optional[Dict[str, Any]]:
        if not (isinstance(image.data, str) and image.data.startswith("data:image")):
            logger.warning("Visual observer skipped invalid image data")
            return None
        return {
            "type": "image_url",
            "image_url": {"url": image.data, "detail": "auto"},
        }

    def _build_question_content(
        self,
        text_prompt: str,
        query_type: VisualQueryType,
        images: List[ImageData],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"视觉任务类型：{query_type.value}。"
                    "后续内容属于同一轮观察，并按标注时间理解。"
                ),
            }
        ]
        for image in images:
            image_block = self._image_block(image)
            if image_block is None:
                continue
            content.append({"type": "text", "text": self._time_label(image)})
            content.append(image_block)

        question = text_prompt or "请客观描述当前最重要的可见内容。"
        content.append(
            {
                "type": "text",
                "text": f"用户当前问题：{question}\n只输出规定的 JSON。",
            }
        )
        return content

    @staticmethod
    def _extract_json(raw_response: str) -> Optional[Dict[str, Any]]:
        stripped = raw_response.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            value = [value] if value else []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _as_confidence(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)

    @classmethod
    def _answerability(
        cls, payload: Dict[str, Any], observations: List[str]
    ) -> str:
        """Normalize new three-level and legacy boolean model outputs."""
        uncertainty_markers = (
            "没看清",
            "无法",
            "不足",
            "未捕捉",
            "没有可用",
            "无可靠",
            "不清楚",
        )
        has_positive_evidence = any(
            not any(marker in item for marker in uncertainty_markers)
            for item in observations
        )
        status = str(payload.get("answerability", "")).strip().lower()
        if status in {"full", "partial", "none"}:
            # A small model may label known evidence as none. Preserve facts.
            if status == "none" and has_positive_evidence:
                return "partial"
            return status
        if cls._as_bool(payload.get("answerable", False)):
            return "full"
        return "partial" if has_positive_evidence else "none"

    @staticmethod
    def _direct_answer(
        status: str, direct_answer: str, observations: List[str]
    ) -> str:
        direct_answer = direct_answer.strip()
        if status == "partial" and (
            not direct_answer
            or "没看清" in direct_answer
            or "无法判断" in direct_answer
            or "无法确认" in direct_answer
        ):
            evidence = "；".join(observations)
            return f"已确认的可见事实：{evidence}。其他细节可能没有完整捕捉。"
        if status == "none" and not direct_answer:
            return "我刚才没有捕捉到可靠的可见事实。"
        return direct_answer

    @classmethod
    def _parse_observation(
        cls, raw_response: str, query_type: VisualQueryType
    ) -> Optional[VisualObservation]:
        payload = cls._extract_json(raw_response)
        if payload is None:
            return None
        observations = cls._as_string_list(payload.get("observations", []))
        answerability = cls._answerability(payload, observations)
        direct_answer = cls._direct_answer(
            answerability,
            str(payload.get("direct_answer", "")),
            observations,
        )
        return VisualObservation(
            query_type=query_type,
            answerable=answerability != "none",
            direct_answer=direct_answer,
            observations=observations,
            confidence=cls._as_confidence(payload.get("confidence", 0.0)),
            answerability=answerability,
        )

    async def _complete(self, messages: List[Dict[str, Any]], system: str) -> str:
        response_parts: List[str] = []
        async with self._llm_lock:
            async for event in self._llm.chat_completion(messages, system):
                if isinstance(event, str):
                    response_parts.append(event)
                elif isinstance(event, dict) and event.get("type") == "text_delta":
                    response_parts.append(str(event.get("text", "")))
        return "".join(response_parts)

    def _apply_scene_payload(
        self,
        scene_payload: Dict[str, Any],
        change_payload: Dict[str, Any],
        captured_at_ms: int,
    ) -> bool:
        """Apply a scene result only if it is not older than current state."""
        if (
            self._scene_state is not None
            and captured_at_ms < self._scene_state.captured_at_ms
        ):
            logger.debug(
                "Ignored stale visual state result: "
                f"{captured_at_ms} < {self._scene_state.captured_at_ms}"
            )
            return False

        self._scene_state = VisualSceneState(
            captured_at_ms=captured_at_ms,
            subject_visible=self._as_bool(
                scene_payload.get("subject_visible", False)
            ),
            scene_summary=str(scene_payload.get("scene_summary", "")).strip(),
            appearance=self._as_string_list(scene_payload.get("appearance", [])),
            objects=self._as_string_list(scene_payload.get("objects", [])),
            current_action=str(scene_payload.get("current_action", "")).strip(),
            expression=str(scene_payload.get("expression", "")).strip(),
            confidence=self._as_confidence(scene_payload.get("confidence", 0.0)),
        )

        description = str(
            change_payload.get("description", change_payload.get("event", ""))
        ).strip()
        if self._as_bool(change_payload.get("changed", False)) and description:
            event = VisualEvent(
                captured_at_ms=captured_at_ms,
                description=description,
                event_type=str(
                    change_payload.get("event_type", "scene_change")
                ).strip(),
                before=str(change_payload.get("before", "")).strip(),
                after=str(change_payload.get("after", "")).strip(),
                confidence=self._as_confidence(
                    change_payload.get("confidence", 0.0)
                ),
            )
            if not self._events or self._events[-1].description != description:
                self._events.append(event)

        logger.info(
            "Continuous visual state updated: "
            f"visible={self._scene_state.subject_visible}, "
            f"confidence={self._scene_state.confidence:.2f}, "
            f"scene={self._scene_state.scene_summary}"
        )
        return True

    async def observe_turn(
        self, text_prompt: str, images: List[ImageData]
    ) -> Optional[VisualObservation]:
        """Use exactly one latest camera frame for state and turn evidence."""
        camera_images = self._ordered_camera_images(images)
        if not camera_images:
            return None
        current = camera_images[-1]
        image_block = self._image_block(current)
        if image_block is None:
            return None

        previous_context = self.get_chat_context() or "上一状态与事件：无。"
        question = text_prompt or "请更新当前场景。"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": previous_context},
                    {"type": "text", "text": f"用户本轮问题：{question}"},
                    {"type": "text", "text": "下面是 ASR 完成后的唯一当前画面："},
                    image_block,
                    {"type": "text", "text": "请只输出规定的 JSON。"},
                ],
            }
        ]
        raw_response = await self._complete(messages, self._TURN_SYSTEM_PROMPT)
        payload = self._extract_json(raw_response)
        if payload is None:
            logger.warning(f"Turn observer returned invalid JSON: {raw_response[:500]}")
            return VisualObservation.unavailable(
                VisualQueryType.CURRENT, "视觉观察结果无法解析"
            )

        scene_payload = payload.get("scene_state", {})
        change_payload = payload.get("scene_change", {})
        if not isinstance(scene_payload, dict):
            scene_payload = {}
        if not isinstance(change_payload, dict):
            change_payload = {}
        captured_at_ms = current.captured_at_ms or int(time.time() * 1000)
        self._apply_scene_payload(scene_payload, change_payload, captured_at_ms)

        question_payload = payload.get("question_observation", {})
        if not isinstance(question_payload, dict) or not self._as_bool(
            question_payload.get("visual_relevance", False)
        ):
            return None

        query_type = (
            VisualQueryType.TEMPORAL
            if any(hint in text_prompt.lower() for hint in self._TEMPORAL_HINTS)
            else VisualQueryType.CURRENT
        )
        observations = self._as_string_list(
            question_payload.get("observations", [])
        )
        answerability = self._answerability(question_payload, observations)
        direct_answer = self._direct_answer(
            answerability,
            str(question_payload.get("direct_answer", "")),
            observations,
        )
        observation = VisualObservation(
            query_type=query_type,
            answerable=answerability != "none",
            direct_answer=direct_answer,
            observations=observations,
            confidence=self._as_confidence(question_payload.get("confidence", 0.0)),
            answerability=answerability,
        )
        logger.info(
            "Unified turn visual observation: "
            f"type={observation.query_type.value}, "
            f"answerability={observation.answerability}, "
            f"confidence={observation.confidence:.2f}"
        )
        return observation

    async def update_scene(self, images: List[ImageData]) -> bool:
        camera_images = self._ordered_camera_images(images)
        if not camera_images:
            return False
        current = camera_images[-1]
        if (
            self.has_fresh_state()
            and current.motion_score is not None
            and current.motion_score < _SCENE_UPDATE_MOTION_THRESHOLD
        ):
            logger.debug(
                "Skipped scene update for unchanged camera frame: "
                f"motion_score={current.motion_score:.3f}"
            )
            return False
        image_block = self._image_block(current)
        if image_block is None:
            return False

        previous_context = self.get_chat_context() or "上一状态：无，这是首次观察。"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": previous_context},
                    {"type": "text", "text": "下面是摄像头当前画面："},
                    image_block,
                    {"type": "text", "text": "请只输出规定的 JSON。"},
                ],
            }
        ]
        raw_response = await self._complete(messages, self._SCENE_SYSTEM_PROMPT)
        payload = self._extract_json(raw_response)
        if payload is None:
            logger.warning(f"Scene observer returned invalid JSON: {raw_response[:500]}")
            return False

        captured_at_ms = current.captured_at_ms or int(time.time() * 1000)
        return self._apply_scene_payload(payload, payload, captured_at_ms)

    async def observe(
        self,
        text_prompt: str,
        images: List[ImageData],
        force_query_type: Optional[VisualQueryType] = None,
    ) -> Optional[VisualObservation]:
        query_type = force_query_type or self.classify_query(text_prompt, images)
        if query_type == VisualQueryType.NONE:
            return None

        selected_images = self.select_images(images, query_type)
        if not selected_images:
            return VisualObservation.unavailable(query_type, "没有可用的视觉内容")

        messages = [
            {
                "role": "user",
                "content": self._build_question_content(
                    text_prompt, query_type, selected_images
                ),
            }
        ]
        raw_response = await self._complete(messages, self._QUESTION_SYSTEM_PROMPT)
        observation = self._parse_observation(raw_response, query_type)
        if observation is None:
            logger.warning(f"Visual observer returned invalid JSON: {raw_response[:500]}")
            return VisualObservation.unavailable(query_type, "视觉观察结果无法解析")

        logger.info(
            "Visual observation: "
            f"type={observation.query_type.value}, "
            f"answerable={observation.answerable}, "
            f"confidence={observation.confidence:.2f}, "
            f"answer={observation.direct_answer}"
        )
        return observation
