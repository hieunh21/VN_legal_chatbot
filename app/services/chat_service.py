from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.session import Session
from app.models.message import Message


async def create_session(db: AsyncSession) -> Session:
    """Create a new chat session."""
    session = Session()
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """Get a session by ID."""
    return await db.get(Session, session_id)


async def get_history(db: AsyncSession, session_id: str) -> list[Message]:
    """Get all messages for a session, ordered by time."""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    return list(result.scalars().all())


async def save_message(db: AsyncSession, session_id: str, role: str, content: str) -> Message:
    """Save a message to the database."""
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message
