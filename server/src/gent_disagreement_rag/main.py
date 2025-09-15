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
