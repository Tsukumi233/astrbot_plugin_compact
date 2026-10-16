import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from astrbot.core.utils.session_lock import session_lock_manager

from main import CompactPlugin


class Manager:
    def __init__(self):
        self.cid = "conversation"
        self.history = []
        for index in range(6):
            self.history.extend(
                [
                    {"role": "user", "content": f"task {index} " + "detail " * 300},
                    {"role": "assistant", "content": "result " * 300},
                ]
            )
        self.writes = []

    async def get_curr_conversation_id(self, umo):
        return self.cid

    async def get_conversation(self, umo, cid):
        return SimpleNamespace(history=json.dumps(self.history))

    async def update_conversation(self, umo, cid, **kwargs):
        self.writes.append((umo, cid, kwargs))
        self.history = kwargs["history"]


class CompactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager()
        self.original = copy.deepcopy(self.manager.history)
        self.provider = SimpleNamespace(
            provider_config={"modalities": ["text"]},
            text_chat=AsyncMock(
                return_value=SimpleNamespace(
                    completion_text="Tasks completed; continue testing."
                )
            ),
        )
        self.options = {
            "provider_id": "summary",
            "keep_recent_ratio": 0.15,
            "instruction": "Preserve decisions.",
        }
        self.config = {
            "agent_runner": {
                "runner_type": "local",
                "config": {"compression": self.options},
            }
        }
        self.context = SimpleNamespace(
            get_config=Mock(return_value=self.config),
            conversation_manager=self.manager,
            get_provider_by_id=Mock(return_value=self.provider),
            get_using_provider_async=AsyncMock(return_value=self.provider),
        )
        self.plugin = CompactPlugin(self.context)

    async def test_native_summary_and_recent_round_preserved(self):
        result = await self.plugin._compact("session")
        self.assertIn("已压缩", result)
        self.assertEqual(len(self.manager.writes), 1)
        self.assertIn("Tasks completed", self.manager.history[0]["content"])
        self.assertEqual(
            self.manager.history[-1]["content"], self.original[-1]["content"]
        )
        self.context.get_config.assert_called_once_with(umo="session")
        self.context.get_provider_by_id.assert_called_once_with("summary")
        self.assertEqual(self.manager.writes[0][2]["token_usage"], 0)

    async def test_empty_summary_preserves_history(self):
        self.provider.text_chat.return_value.completion_text = " "
        await self.plugin._compact("session")
        self.assertEqual(self.manager.history, self.original)
        self.assertFalse(self.manager.writes)

    async def test_model_failure_never_truncates(self):
        self.provider.text_chat.side_effect = RuntimeError("synthetic failure")
        await self.plugin._compact("session")
        self.assertFalse(self.manager.writes)

    async def test_larger_summary_not_saved(self):
        self.provider.text_chat.return_value.completion_text = "larger " * 10000
        await self.plugin._compact("session")
        self.assertFalse(self.manager.writes)

    async def test_changed_history_not_overwritten(self):
        async def reset(**kwargs):
            self.manager.history = []
            return SimpleNamespace(completion_text="summary")

        self.provider.text_chat.side_effect = reset
        result = await self.plugin._compact("session")
        self.assertIn("已放弃", result)
        self.assertEqual(self.manager.history, [])
        self.assertFalse(self.manager.writes)

    async def test_switched_conversation_not_overwritten(self):
        async def switch(**kwargs):
            self.manager.cid = "new-conversation"
            return SimpleNamespace(completion_text="summary")

        self.provider.text_chat.side_effect = switch
        await self.plugin._compact("session")
        self.assertFalse(self.manager.writes)

    async def test_current_provider_fallback(self):
        self.options["provider_id"] = ""
        await self.plugin._compact("session")
        self.context.get_using_provider_async.assert_awaited_once_with("session")

    async def test_checkpoint_and_tool_round_survive(self):
        self.manager.history.extend(
            [
                {"role": "user", "content": "latest task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call1", "content": "tool result"},
                {"role": "assistant", "content": "latest answer"},
                {"role": "_checkpoint", "content": {"id": "latest-checkpoint"}},
            ]
        )
        await self.plugin._compact("session")
        self.assertEqual(
            self.manager.history[-1]["content"], {"id": "latest-checkpoint"}
        )
        self.assertTrue(
            any(item.get("tool_call_id") == "call1" for item in self.manager.history)
        )

    async def test_cancelled_request_releases_lock_and_pending(self):
        started = asyncio.Event()

        async def slow(**kwargs):
            started.set()
            await asyncio.Event().wait()

        self.provider.text_chat.side_effect = slow
        event = SimpleNamespace(
            unified_msg_origin="session",
            should_call_llm=Mock(),
            plain_result=lambda text: text,
            send=AsyncMock(),
        )

        async def run():
            return [result async for result in self.plugin.compact(event)]

        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.plugin._pending)
        self.assertFalse(self.manager.writes)
        async with asyncio.timeout(1):
            async with session_lock_manager.acquire_lock("session"):
                pass

    async def test_zero_recent_ratio_keeps_latest_checkpoint(self):
        self.options["keep_recent_ratio"] = 0
        self.manager.history.append(
            {"role": "_checkpoint", "content": {"id": "latest"}}
        )
        await self.plugin._compact("session")
        self.assertEqual(self.manager.history[-1]["content"], {"id": "latest"})

    async def test_missing_explicit_provider_does_not_fallback(self):
        self.context.get_provider_by_id.return_value = None
        await self.plugin._compact("session")
        self.context.get_using_provider_async.assert_not_awaited()
        self.assertFalse(self.manager.writes)

    async def test_empty_and_external_sessions_do_not_call_model(self):
        self.manager.history = []
        await self.plugin._compact("session")
        self.config["agent_runner"]["runner_type"] = "dify"
        await self.plugin._compact("session")
        self.provider.text_chat.assert_not_awaited()

    async def test_command_waits_for_core_session_lock_and_rejects_duplicate(self):
        event = SimpleNamespace(
            unified_msg_origin="session",
            should_call_llm=Mock(),
            plain_result=lambda text: text,
            send=AsyncMock(),
        )

        async def run():
            return [result async for result in self.plugin.compact(event)]

        async with session_lock_manager.acquire_lock("session"):
            task = asyncio.create_task(run())
            await asyncio.sleep(0)
            event.send.assert_awaited_once_with("开始压缩")
            self.provider.text_chat.assert_not_awaited()
            duplicate = await run()
            self.assertIn("已有压缩任务", duplicate[0])
            event.send.assert_awaited_once_with("开始压缩")
        result = await task
        self.assertIn("已压缩", result[0])
        self.assertFalse(self.plugin._pending)


if __name__ == "__main__":
    unittest.main()
