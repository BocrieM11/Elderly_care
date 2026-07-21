import json
import time
import unittest

from open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from open_llm_vtuber.agent.input_types import (
    BatchInput,
    ImageData,
    ImageSource,
    TextData,
    TextSource,
)
from open_llm_vtuber.agent.visual_observer import (
    VisualObserver,
    VisualQueryType,
)


def image(source: ImageSource, *, motion_score: float = 0.1) -> ImageData:
    return ImageData(
        source=source,
        data="data:image/jpeg;base64,AA==",
        mime_type="image/jpeg",
        captured_at_ms=int(time.time() * 1000),
        motion_score=motion_score,
    )


class FakeVisionLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, messages, system):
        self.calls += 1
        yield json.dumps(
            {
                "subject_visible": True,
                "scene_summary": "用户坐在镜头前",
                "appearance": [],
                "objects": [],
                "current_action": "坐着",
                "expression": "",
                "confidence": 0.9,
                "changed": False,
                "event": "",
            },
            ensure_ascii=False,
        )


class FakeObserver:
    def __init__(self) -> None:
        self.observe_calls = 0
        self.context_calls = 0

    def classify_query(self, text_prompt, images):
        return VisualObserver.classify_query(text_prompt, images)

    async def observe_turn(self, text_prompt, images):
        self.observe_calls += 1
        return None

    def get_chat_context(self):
        self.context_calls += 1
        return "scene-context"


def make_agent(observer: FakeObserver) -> BasicMemoryAgent:
    agent = object.__new__(BasicMemoryAgent)
    agent._memory = []
    agent._visual_observer = observer
    return agent


def batch(text: str, images=None) -> BatchInput:
    return BatchInput(
        texts=[TextData(source=TextSource.INPUT, content=text)],
        images=images,
    )


class VisualQueryClassificationTests(unittest.TestCase):
    def test_plain_camera_turn_is_not_a_visual_query(self):
        result = VisualObserver.classify_query(
            "给我讲个故事", [image(ImageSource.CAMERA)]
        )
        self.assertEqual(result, VisualQueryType.NONE)

    def test_explicit_visual_and_temporal_phrases_are_detected(self):
        self.assertEqual(
            VisualObserver.classify_query("看看画面里有什么", []),
            VisualQueryType.CURRENT,
        )
        self.assertEqual(
            VisualObserver.classify_query("我刚才做了什么", []),
            VisualQueryType.TEMPORAL,
        )

    def test_media_without_a_visual_question_does_not_start_observer(self):
        result = VisualObserver.classify_query(
            "帮我保存这个", [image(ImageSource.UPLOAD)]
        )
        self.assertEqual(result, VisualQueryType.NONE)


class SceneUpdateThrottleTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_motion_is_skipped_only_after_initial_state(self):
        llm = FakeVisionLLM()
        observer = VisualObserver(llm)

        first = image(ImageSource.CAMERA, motion_score=0.01)
        self.assertTrue(await observer.update_scene([first]))
        self.assertEqual(llm.calls, 1)

        unchanged = image(ImageSource.CAMERA, motion_score=0.01)
        unchanged.captured_at_ms = first.captured_at_ms + 1
        self.assertFalse(await observer.update_scene([unchanged]))
        self.assertEqual(llm.calls, 1)

        changed = image(ImageSource.CAMERA, motion_score=0.05)
        changed.captured_at_ms = first.captured_at_ms + 2
        self.assertTrue(await observer.update_scene([changed]))
        self.assertEqual(llm.calls, 2)


class AgentVisualGatingTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_chat_does_not_observe_or_inject_scene(self):
        observer = FakeObserver()
        agent = make_agent(observer)

        messages = await agent._to_messages(
            batch("给我讲个故事", [image(ImageSource.CAMERA)])
        )

        self.assertEqual(observer.observe_calls, 0)
        self.assertEqual(observer.context_calls, 0)
        self.assertEqual(
            messages[-1]["content"],
            [{"type": "text", "text": "给我讲个故事"}],
        )

    async def test_explicit_visual_question_observes_and_injects_scene(self):
        observer = FakeObserver()
        agent = make_agent(observer)

        messages = await agent._to_messages(
            batch("看看画面里有什么", [image(ImageSource.CAMERA)])
        )

        self.assertEqual(observer.observe_calls, 1)
        self.assertEqual(observer.context_calls, 1)
        self.assertEqual(messages[-1]["content"][0]["text"], "scene-context")

    async def test_uploaded_media_still_reaches_chat_model_directly(self):
        observer = FakeObserver()
        agent = make_agent(observer)

        messages = await agent._to_messages(
            batch("帮我保存这个", [image(ImageSource.UPLOAD)])
        )

        self.assertEqual(observer.observe_calls, 0)
        self.assertEqual(observer.context_calls, 0)
        self.assertEqual(messages[-1]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
