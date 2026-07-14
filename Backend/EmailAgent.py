import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import json

# 将项目根目录（mail-friend/）加入 Python 路径
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import settings
from tool import authenticate, check_inbox, send_email, AuthenticatedState
from Middleware import dynamic_tool_call, dynamic_prompt_fn

load_dotenv(find_dotenv())


def _serialize(obj):
    """递归转换对象为可 JSON 序列化的格式"""
    if hasattr(obj, 'value'):
        return _serialize(obj.value)
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class EmailAgent():
    def __init__(self):
        self.checkpointer = InMemorySaver()
        self.agent = None

    async def init_agent(self):
        self.agent = create_agent(
            "deepseek-chat",
            tools=[authenticate, check_inbox, send_email],
            state_schema=AuthenticatedState,
            checkpointer=self.checkpointer,
            middleware=[
                dynamic_tool_call,
                dynamic_prompt_fn,
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "authenticate": False,
                        "check_inbox": False,
                        "send_email": True,
                    }
                )
            ],
        )

    async def init(self):
        await self.init_agent()

    async def close(self):
        pass

    """
    ####################################辅助函数######################################################
    """

    def get_messages(self, thread_id: str) -> dict:
        """获取会话历史，如果存在中断则返回中断信息"""

        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = self.agent.get_state(config)
        except Exception:
            return {"messages": []}
        if state is None or not state.values:
            return {"messages": []}

        messages = state.values.get("messages", [])

        # 转换消息格式
        result = []
        for msg in messages:
            if not msg.content:
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})

        response = {"messages": result}

        # 检查是否存在中断
        interrupts = None
        if hasattr(state, 'interrupts') and state.interrupts:
            interrupts = state.interrupts
        elif hasattr(state, 'tasks') and state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    interrupts = task.interrupts
                    break

        if interrupts:
            response["has_interrupt"] = True
            response["interrupt"] = {
                "reason": "需要人工确认",
                "details": _serialize(interrupts)
            }

        return response

    def generate_sse(self, thread_id: str, message: str, interrupt_decision: dict):
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
            for chunk in self.agent.stream(
                    _input,
                    config=config,
                    stream_mode=["messages", "updates"],
                    version="v2"
            ):
                event_type = chunk["type"]
                data = chunk["data"]

                # 1. messages - LLM 流式输出（token 级别）
                if event_type == "messages":
                    token, metadata = data
                    content = None
                    if isinstance(token, AIMessage) and hasattr(token, "content"):
                        content = token.content

                    if content:
                        yield f"data: {json.dumps({'type': 'message', 'content': content}, ensure_ascii=False)}\n\n"

                # 2. updates - 节点完成的 state 更新
                elif event_type == "updates":
                    if "__interrupt__" in data:
                        interrupt_data = data['__interrupt__']
                        details = _serialize(interrupt_data)
                        yield f"data: {json.dumps({'type': 'interrupt', 'interrupt': {'reason': '需要人工确认', 'details': details}}, ensure_ascii=False, default=str)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'content': '处理完成'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"


email_agent = EmailAgent()
__all__ = ["email_agent"]
