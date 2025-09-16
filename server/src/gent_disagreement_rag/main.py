from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .core import RAGService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to gent_disagreement_rag"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/chat")
async def chat(request: Request):
    data = await request.json()
    user_text = data["versions"][0]["content"]
    rag_service = RAGService()
    response = rag_service.ask_question(user_text)
    return {"message": response}


@app.post("/api/chat")
async def chat_ai_sdk(request: Request):
    """AI SDK compatible chat endpoint"""
    data = await request.json()
    messages = data.get("messages", [])

    # Extract the latest user message from AI SDK format
    if not messages:
        return {"error": "No messages provided"}

    # Get the last user message
    last_message = messages[-1]
    if last_message.get("role") != "user":
        return {"error": "Last message must be from user"}

    # Extract text content from either parts (default format) or text (text streaming format)
    user_text = ""

    # Check for text streaming format first
    if "text" in last_message:
        user_text = last_message.get("text", "")
    else:
        # Fall back to parts format
        parts = last_message.get("parts", [])
        for part in parts:
            if part.get("type") == "text":
                user_text += part.get("text", "")

    if not user_text.strip():
        return {"error": "No text content found in user message"}

    rag_service = RAGService()

    # Use simple text streaming compatible with AI SDK
    return StreamingResponse(
        content=rag_service.ask_question_text_stream(user_text),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/v1/chat/stream")
async def stream_chat(request: Request):
    data = await request.json()
    user_text = data["versions"][0]["content"]
    rag_service = RAGService()

    return StreamingResponse(
        content=rag_service.ask_question(user_text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# poetry run uvicorn gent_disagreement_rag.main:app
