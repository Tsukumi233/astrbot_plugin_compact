"""Manually invoke AstrBot's native summary compressor for the current session."""

import asyncio
import json

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.agent.context.compressor import LLMSummaryCompressor
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
from astrbot.core.agent.message import (
    bind_checkpoint_messages,
    dump_messages_with_checkpoints,
)
from astrbot.core.utils.session_lock import session_lock_manager


class CompactPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self._pending: set[str] = set()

    @filter.command("compact")
    @filter.permission_type(filter.PermissionType.SHARED_GROUP_ADMIN)
    async def compact(self, event: AstrMessageEvent):
        """立即压缩当前会话，保留摘要和最近上下文。"""
        event.should_call_llm(False)
        umo = event.unified_msg_origin
        if umo in self._pending:
            yield event.plain_result("当前会话已有压缩任务，请稍候。")
            return
        self._pending.add(umo)
        try:
            # Include lock waiting in the timeout. Never interrupt another reply.
            async with asyncio.timeout(180):
                async with session_lock_manager.acquire_lock(umo):
                    result = await self._compact(umo)
        except TimeoutError:
            result = "压缩等待或请求超时，未主动替换会话历史，请稍后重试。"
        except Exception as exc:  # noqa: BLE001 - contain provider/plugin failures
            # Provider errors may contain credentials or conversation text.
            logger.warning("compact failed (%s)", type(exc).__name__)
            result = "压缩失败，未主动清空或截断历史，请检查模型配置后重试。"
        finally:
            self._pending.discard(umo)
        yield event.plain_result(result)

    async def _compact(self, umo: str) -> str:
        config = self.context.get_config(umo=umo)
        runner = config.get("agent_runner", {})
        if runner.get("runner_type") != "local":
            return "仅支持 AstrBot 内置 Agent；第三方 Agent 的历史由对应服务管理。"
        manager = self.context.conversation_manager
        cid = await manager.get_curr_conversation_id(umo)
        if not cid:
            return "当前没有可压缩的会话。"
        conversation = await manager.get_conversation(umo, cid)
        if not conversation:
            return "当前会话不存在。"
        original = json.loads(conversation.history or "[]")
        messages = bind_checkpoint_messages(original)
        if not any(message.role != "system" for message in messages):
            return "当前会话没有可压缩的历史。"
        options = runner.get("config", {}).get("compression", {})
        provider_id = options.get("provider_id", "")
        provider = (
            self.context.get_provider_by_id(provider_id)
            if provider_id
            else await self.context.get_using_provider_async(umo)
        )
        if provider is None:
            return "压缩模型不可用，请检查当前配置文件的压缩模型或聊天模型。"
        counter = EstimateTokenCounter()
        before = counter.count_tokens(messages)
        compressor = LLMSummaryCompressor(
            provider=provider,
            keep_recent_ratio=options.get("keep_recent_ratio", 0.15),
            instruction_text=options.get("instruction", ""),
            token_counter=counter,
        )
        # Intentionally bypass should_compress(): /compact is an explicit request.
        compressed = await compressor(messages)
        after = counter.count_tokens(compressed)
        if compressed == messages or after >= before:
            return (
                "未获得更短的有效摘要，保留原始历史（模型可能失败，或历史已足够短）。"
            )
        # A zero recent ratio can summarize the final checkpoint away. Keep the
        # latest platform-history cursor so context adapters do not replay it.
        checkpoints = [m._checkpoint_after for m in messages if m._checkpoint_after]
        if checkpoints and not any(
            m._checkpoint_after == checkpoints[-1] for m in compressed
        ):
            compressed[-1]._checkpoint_after = checkpoints[-1]
        # /reset and /new do not take the Agent lock. Recheck after the LLM await.
        current_id = await manager.get_curr_conversation_id(umo)
        latest = await manager.get_conversation(umo, cid)
        if (
            current_id != cid
            or latest is None
            or json.loads(latest.history or "[]") != original
        ):
            return "压缩期间会话已切换或历史已变化，已放弃本次结果，请重试。"
        await manager.update_conversation(
            umo, cid, history=dump_messages_with_checkpoints(compressed), token_usage=0
        )
        return f"✅ 已压缩当前会话：约 {before:,} → {after:,} Token。保留摘要及近期上下文。"
