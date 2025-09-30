from fastapi import FastAPI
from app.routers import classify

app = FastAPI(title="Zero-Shot LLM Text Classification API", version="1.0")

app.include_router(classify.router, prefix="/api/v1", tags=["Classification"])
