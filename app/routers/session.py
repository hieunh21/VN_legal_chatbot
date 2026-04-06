from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.services import chat_service
from app.models.schemas import SessionOut, MessageOut

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut)
async def create_session(db: AsyncSession = Depends(get_db)):
    """Create a new chat session."""
    session = await chat_service.create_session(db)
    return session


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get session with message history."""
    session = await chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await chat_service.get_history(db, session_id)
    return SessionOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        messages=[MessageOut(**m.__dict__) for m in messages],
    )
