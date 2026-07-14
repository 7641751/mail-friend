from typing import Optional, Dict, Any, List

import json

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from pydantic import BaseModel
from config import settings
from langgraph.types import Command

class ChatRequest(BaseModel):
    message: Optional[str] = None
    image_url: Optional[str] = None
    thread_id: str
    # 用户确认的interrupt操作
    interrupt_decision: Optional[Dict[str, Any]] = None

def _serialize_interrupt(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_interrupt(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_interrupt(v) for k, v in value.items()}
    return value

async def generate_sse(self, thread_id: str, message: str, interrupt_decision: dict):
    """生成 SSE 事件流"""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # 使用 HumanMessage 封装用户消息
    messages = {"messages": [HumanMessage(content=message)]}
    # 判断是消息还是 command
    _input = messages
    if interrupt_decision:
        _input = Command(resume={
            "decisions": [interrupt_decision]
        })


    try:
        async for event in self.agent.astream_events(
            _input,
            config=config,
            version="v2"
        ):
            kind = event["event"]

            # 1. on_chat_model_stream - LLM 流式输出（token 级别）
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"type": "message", "content": chunk.content},
                            ensure_ascii=False
                        )
                    }

            # 2. on_chain_end - 节点完成的 state 更新（含中断）
            elif kind == "on_chain_end":
                output = event["data"].get("output", {})
                if isinstance(output, dict) and "__interrupt__" in output:
                    interrupt_data = output["__interrupt__"]
                    details = _serialize_interrupt(interrupt_data)
                    yield {
                        "event": "interrupt",
                        "data": json.dumps(
                            {
                                "type": "interrupt",
                                "interrupt": {
                                    "reason": "需要人工确认",
                                    "details": details
                                }
                            },
                            ensure_ascii=False, default=str
                        )
                    }

        yield {
            "event": "done",
            "data": json.dumps({"type": "done", "content": "处理完成"}, ensure_ascii=False)
        }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False)
        }
