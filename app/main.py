from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import session, chat


from app.vector_store import qdrant_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: models are loaded on first import
    print("Server started. Models will load on first request.")
    qdrant_store.ensure_collection()
    yield
    # Shutdown
    print("Server stopped.")


app = FastAPI(title="VN Legal RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {"message": "VN Legal RAG Chatbot API"}
