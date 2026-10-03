"""
AI Safety & Evaluation Gateway for Healthcare Assistants.
Main application runner and FastAPI app export.
"""
import uvicorn
from app.main import app
from app.config import settings

if __name__ == "__main__":
    print(f"Starting AI Safety Gateway on http://{settings.HOST}:{settings.PORT}")
    print(f"API Documentation available at: http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
