from fastapi import FastAPI
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
async def chat():
    rag_service = RAGService()
    response = rag_service.ask_question("Was Elon Musk mentioned?")
    return {"message": response}


# poetry run uvicorn gent_disagreement_rag.main:app
