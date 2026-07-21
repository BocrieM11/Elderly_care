import unittest

from companion_cn.api import _AVATAR_ACTIONS, _choose_avatar_action


class AvatarActionTests(unittest.TestCase):
    def test_distress_overrides_playful_words(self):
        self.assertEqual(
            _choose_avatar_action("我们以后再跳舞。", "我今天很孤单", "loneliness", 0.8),
            "big_comfort",
        )

    def test_wave_greeting_is_immediate_welcome(self):
        self.assertEqual(
            _choose_avatar_action("您好呀。", "用户刚刚主动向你挥手打招呼", "neutral", 0.2),
            "welcome",
        )

    def test_specific_exercise_action(self):
        self.assertEqual(
            _choose_avatar_action("咱们做几下。", "想练原地踏步", "neutral", 0.2),
            "march",
        )

    def test_every_result_has_a_bundled_asset(self):
        cases = (
            ("晚安", "neutral", 0.2),
            ("谢谢你", "neutral", 0.2),
            ("今天真高兴", "joy", 0.8),
            ("普通聊天", "neutral", 0.2),
        )
        for text, emotion, intensity in cases:
            with self.subTest(text=text):
                self.assertIn(
                    _choose_avatar_action("", text, emotion, intensity),
                    _AVATAR_ACTIONS,
                )


if __name__ == "__main__":
    unittest.main()
