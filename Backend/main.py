import json
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from EmailAgent import email_agent


# ========== 会话存储 ==========
sessions: dict = {}


# ========== 请求模型 ==========

class CreateSessionRequest(BaseModel):
    user_id: str
    biz_type: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    message: Optional[str] = None
    thread_id: str
    interrupt_decision: Optional[dict] = None


# ========== 应用生命周期 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    await email_agent.init()
    yield
    await email_agent.close()


app = FastAPI(
    title="MailFriend API",
    description="邮件助手智能体 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 会话管理 API ==========

@app.get("/api/v1/sessions")
async def list_sessions(user_id: str = Query(...), biz_type: str = Query(...)):
    """获取会话列表"""
    result = [
        s for s in sessions.values()
        if s["user_id"] == user_id and s["biz_type"] == biz_type
    ]
    return sorted(result, key=lambda x: x["created_at"], reverse=True)


@app.post("/api/v1/sessions")
async def create_session(req: CreateSessionRequest):
    """创建新会话"""
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "thread_id": thread_id,
        "user_id": req.user_id,
        "biz_type": req.biz_type,
        "name": req.name or f"新会话 {len(sessions) + 1}",
        "created_at": now,
    }
    sessions[thread_id] = session
    return session


@app.delete("/api/v1/sessions/{thread_id}")
async def delete_session(thread_id: str):
    """删除会话"""
    if thread_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    del sessions[thread_id]
    return {"ok": True}


# ========== 消息 API ==========

@app.get("/api/v1/sessions/{thread_id}/messages")
async def get_messages(thread_id: str):
    """获取会话消息历史"""
    if thread_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    return email_agent.get_messages(thread_id)


# ========== 聊天 API（SSE 流式） ==========

@app.post("/api/v1/chat/send")
async def send_message(req: ChatRequest):
    """发送消息并返回 SSE 流"""
    return StreamingResponse(
        email_agent.generate_sse(req.thread_id, req.message or "", req.interrupt_decision),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ========== 静态文件 ==========

app.mount("/", StaticFiles(directory="../Frontend", html=True), name="frontend")
