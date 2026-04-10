import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import async_session, get_db
from app.models.schemas import ChatRequest, ChatResponse
from app.services import chat_service, rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Non-streaming endpoint — preserved for compatibility."""
    session = await chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = await chat_service.get_history(db, session_id)
    result = rag_service.answer(request.query, history)

    await chat_service.save_message(db, session_id, "user", request.query)
    await chat_service.save_message(db, session_id, "assistant", result["answer"])

    return ChatResponse(answer=result["answer"], sources=result["sources"])

@router.post("/{session_id}/stream")
async def chat_stream(session_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Streaming endpoint via Server-Sent Events."""
    session = await chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = await chat_service.get_history(db, session_id)
    await chat_service.save_message(db, session_id, "user", request.query)

    collected: list[str] = []

    async def event_generator():
        try:
            for chunk in rag_service.answer_stream(request.query, history):
                if chunk not in ("data: [DONE]\n\n",):
                    try:
                        data = json.loads(chunk[6:])  # strip "data: "
                        if data.get("type") == "token":
                            collected.append(data["content"])
                    except (json.JSONDecodeError, KeyError):
                        pass
                yield chunk
        except asyncio.CancelledError:
            pass  # stream bị client ngắt (ví dụ user đóng tab)
        finally:
            if collected:
                # Dùng một session DB mới vì session cũ (từ Depends) có thể đã bị đóng khi cancel
                async with async_session() as new_db:
                    await chat_service.save_message(new_db, session_id, "assistant", "".join(collected))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

