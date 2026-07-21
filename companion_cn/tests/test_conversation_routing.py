import unittest
from datetime import datetime
from unittest.mock import patch

from companion_cn.api import ReminderCreateRequest, create_reminder
from companion_cn.nodes import (
    _extract_reminder_content,
    _personal_management_tool,
    build_persona,
    context_assemble,
    emotion_detect,
    ground_visual_response,
    remove_virtual_actions,
)


def make_state(user_input, messages, *, with_image=True):
    return {
        "user_id": "test-user",
        "session_id": "test-session",
        "user_input": user_input,
        "messages": messages,
        "visual_images": (
            [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}}]
            if with_image else []
        ),
        "risk_level": 0,
        "risk_label": None,
        "emotion": None,
        "memory_facts": [],
        "system_prompt": "",
        "response": None,
        "response_cleaned": None,
        "error": None,
        "temperature": 0.75,
        "topic_summary": "",
        "asked_questions": [],
        "dialect": "northern",
        "tool_result": None,
        "weather_city": "北京",
    }


class ReminderRoutingTests(unittest.TestCase):
    def test_natural_relative_reminder_extracts_only_the_task(self):
        staged = {
            "id": 12,
            "content": "吃药",
            "due_at": "2026-07-21T12:00:00+08:00",
        }
        with patch("companion_cn.nodes.create_pending", return_value=staged) as create_pending:
            result = _personal_management_tool("test-user", "一分钟后提醒我吃药")

        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["content"], "吃药")
        self.assertEqual(_extract_reminder_content("一分钟后提醒我吃药"), "吃药")
        self.assertEqual(create_pending.call_args.args[1], "吃药")

    def test_visual_food_question_is_not_a_reminder_completion(self):
        with patch("companion_cn.nodes.complete_latest") as complete_latest:
            result = _personal_management_tool("test-user", "看我表情猜猜我吃了啥")

        self.assertIsNone(result)
        complete_latest.assert_not_called()

    def test_actual_completion_report_still_completes_latest_reminder(self):
        with patch(
            "companion_cn.nodes.complete_latest",
            return_value={"content": "吃药"},
        ) as complete_latest:
            result = _personal_management_tool("test-user", "我已经吃过药了")

        self.assertEqual(result["type"], "reminder")
        self.assertIn("吃药", result["answer"])
        complete_latest.assert_called_once_with("test-user")


class ReminderApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_repeat_rule_is_normalized(self):
        request = ReminderCreateRequest(
            user_id="test-user",
            content="吃药",
            due_at=datetime(2026, 7, 22, 8, 0),
            repeat_rule=None,
        )
        staged = {"id": 7, "content": "吃药", "due_at": "2026-07-22T08:00:00+08:00"}
        with patch("companion_cn.api.create_pending", return_value=staged) as create_pending, patch(
            "companion_cn.api.activate_reminder", return_value=staged
        ):
            result = await create_reminder(request)

        self.assertTrue(result["ok"])
        self.assertEqual(create_pending.call_args.args[3], "")


class PersonaTests(unittest.TestCase):
    def test_persona_is_a_respectful_companion_without_big_sister_identity(self):
        persona = build_persona("northern")

        self.assertIn("养老院场景中的情感陪伴助手", persona)
        self.assertIn("平等、自然的成人对话", persona)
        self.assertIn("始终只用“我”自称", persona)
        self.assertIn("不要自称大姐、阿姨", persona)
        self.assertIn("我是陪您聊天的智能助手", persona)
        self.assertIn("不要用“乖、听话、真棒、老人家、您这年纪”", persona)
        self.assertNotIn("像邻家热心大姐", persona)
        self.assertNotIn("对方难过你先叹口气", persona)


class CompanionSafetyTests(unittest.TestCase):
    def test_absent_family_member_is_treated_as_loneliness(self):
        result = emotion_detect({"user_input": "孙女一直没来看我"})

        self.assertEqual(result["emotion"]["primary"], "loneliness")
        self.assertEqual(result["emotion_source"], "negated_family_rule")

    def test_physical_action_claim_is_rewritten(self):
        reply = remove_virtual_actions("我马上给您倒杯水，您先歇会儿。")

        self.assertNotIn("我马上给您倒", reply)
        self.assertIn("您先照顾好自己", reply)


class VisualGroundingTests(unittest.TestCase):
    def test_nonvisual_greeting_does_not_send_camera_frame_to_llm(self):
        state = make_state(
            "你好",
            [{"role": "user", "content": "你好"}],
        )

        result = context_assemble(state)

        self.assertEqual(result["messages"][-1]["content"], "你好")
        self.assertEqual(result["visual_images"], [])
        self.assertNotIn("你会看到一张", result["system_prompt"])
        self.assertIn("不是对方发送或上传的照片", result["system_prompt"])
        self.assertIn("不要主动猜测或谈论时段、天气、地点", result["system_prompt"])

    def test_explicit_visual_question_uses_live_camera_without_photo_wording(self):
        state = make_state(
            "你看我举了几根手指",
            [{"role": "user", "content": "你看我举了几根手指"}],
        )

        result = context_assemble(state)

        self.assertIsInstance(result["messages"][-1]["content"], list)
        self.assertEqual(len(result["visual_images"]), 1)
        self.assertEqual(result["messages"][-1]["content"][1]["type"], "image_url")
        self.assertIn("实时视频通话", result["system_prompt"])
        self.assertIn("禁止提到图片、照片", result["system_prompt"])

    def test_now_followup_rechecks_current_image_instead_of_copying_number(self):
        state = make_state(
            "现在呢",
            [
                {"role": "user", "content": "你看我手势是几"},
                {"role": "assistant", "content": "看起来是三。"},
                {"role": "user", "content": "现在呢"},
            ],
        )

        result = context_assemble(state)
        system_prompt = result["system_prompt"]
        current_text = result["messages"][-1]["content"][0]["text"]

        self.assertIn("全新的证据重新判断", system_prompt)
        self.assertIn("不得因为上一轮回答过某个数字", system_prompt)
        self.assertIn("忽略上一轮的视觉数字或结论", current_text)
        self.assertNotIn(
            "看起来是三。",
            [message.get("content") for message in result["messages"]],
        )

    def test_environment_review_forbids_claiming_the_office_is_quiet(self):
        state = make_state(
            "评价一下办公室环境",
            [{"role": "user", "content": "评价一下办公室环境"}],
        )

        result = context_assemble(state)
        system_prompt = result["system_prompt"]
        current_text = result["messages"][-1]["content"][0]["text"]

        self.assertIn("静态画面不能证明安静或嘈杂", system_prompt)
        self.assertIn("只评价可见的空间、采光、整洁和陈设", system_prompt)
        self.assertIn("不要从静态画面推断声音", current_text)

        grounded = ground_visual_response(
            "办公室看着宽敞明亮，但太安静了。",
            state,
        )
        self.assertNotIn("太安静", grounded)
        self.assertIn("仅凭画面无法判断", grounded)

    def test_generic_now_after_nonvisual_topic_is_not_forced_into_visual_followup(self):
        state = make_state(
            "现在呢",
            [
                {"role": "user", "content": "北京天气怎么样"},
                {"role": "assistant", "content": "今天晴。"},
                {"role": "user", "content": "现在呢"},
            ],
        )

        result = context_assemble(state)
        self.assertNotIn("全新的证据重新判断", result["system_prompt"])


if __name__ == "__main__":
    unittest.main()
