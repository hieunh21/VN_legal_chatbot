from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.services import chat_service, rag_service
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Send a query and get RAG-powered response."""
    # Check session exists
    session = await chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get history
    history = await chat_service.get_history(db, session_id)

    # RAG pipeline
    result = rag_service.answer(request.query, history)

    # Save messages
    await chat_service.save_message(db, session_id, "user", request.query)
    await chat_service.save_message(db, session_id, "assistant", result["answer"])

    return ChatResponse(answer=result["answer"], sources=result["sources"])
