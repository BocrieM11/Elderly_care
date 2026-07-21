import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from companion_cn.chat_record_reporter import (
    build_session_payload,
    init_report_table,
    report_session_sync,
)


class _Response:
    def raise_for_status(self):
        return None


class ChatRecordReporterTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE conversations_cn (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_type TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    user_message TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO conversations_cn (
                    session_id, prompt_type, round_number, user_message,
                    reply, created_at
                ) VALUES (?, 'cn', ?, ?, ?, ?)
                """,
                [
                    (
                        "session-1",
                        1,
                        "我今天心情不太好",
                        "怎么了？跟我说说吧",
                        "2026-07-16T06:30:00+00:00",
                    ),
                    (
                        "session-1",
                        2,
                        "就是想家里人了",
                        "想他们的时候可以做点开心的事",
                        "2026-07-16T06:31:00+00:00",
                    ),
                ],
            )
            conn.commit()
        init_report_table(self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_builds_receiver_schema_with_chinese_roles(self):
        payload = build_session_payload(
            "session-1",
            db_path=self.db_path,
            resident_name="张奶奶",
            room_no="302",
        )

        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["timestamp"], "2026-07-16T14:30:00+08:00")
        self.assertEqual(payload["resident_name"], "张奶奶")
        self.assertEqual(payload["room_no"], "302")
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["老人", "AI", "老人", "AI"],
        )

    def test_success_is_persistently_idempotent(self):
        with patch(
            "companion_cn.chat_record_reporter.requests.post",
            return_value=_Response(),
        ) as post:
            first = report_session_sync(
                "session-1",
                db_path=self.db_path,
                endpoint="http://analysis.test/api/chat/records",
                max_retries=0,
                api_token="test-service-token",
            )
            second = report_session_sync(
                "session-1",
                db_path=self.db_path,
                endpoint="http://analysis.test/api/chat/records",
                max_retries=0,
                api_token="test-service-token",
            )

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(post.call_count, 1)
        sent_json = post.call_args.kwargs["json"]
        self.assertEqual(sent_json["messages"][0]["role"], "老人")
        self.assertEqual(
            post.call_args.kwargs["headers"]["X-Service-Token"],
            "test-service-token",
        )
        self.assertNotIn("X-Chat-Record-Token", post.call_args.kwargs["headers"])
        with closing(sqlite3.connect(self.db_path)) as conn:
            status = conn.execute(
                "SELECT status FROM chat_record_reports WHERE session_id = ?",
                ("session-1",),
            ).fetchone()[0]
        self.assertEqual(status, "sent")


if __name__ == "__main__":
    unittest.main()
