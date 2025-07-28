from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Welcome to gent_disagreement_rag"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# poetry run uvicorn main:app
