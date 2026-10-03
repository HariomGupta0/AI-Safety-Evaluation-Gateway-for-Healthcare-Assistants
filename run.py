import argparse
import uvicorn
from app.config import settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the AI Safety Gateway Server")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind server")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting AI Safety Gateway on http://{args.host}:{args.port}")
    print(f"Interactive Swagger Docs available at http://{args.host}:{args.port}/docs")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
