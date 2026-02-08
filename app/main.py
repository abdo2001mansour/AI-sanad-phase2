from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.config.settings import settings
from app.routes.ocr import router as ocr_router
from app.routes.chat import router as chat_router
from app.routes.rag import router as rag_router
from app.routes.rag_user_files import router as rag_user_files_router
from app.routes.rag_database import router as rag_database_router
from app.routes.speech import router as speech_router
from app.routes.s3 import router as s3_router
from app.routes.magic_tools import router as magic_tools_router
from app.routes.meeting_summary import router as meeting_summary_router
from app.routes.legal_tools import router as legal_tools_router
from app.logging_config import configure_logging

# Configure logging to suppress socket errors from client disconnections
configure_logging()

# Create FastAPI instance
app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI OCR Service with Google Gemini AI and Chat Completion",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(ocr_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(rag_router, prefix=settings.API_V1_STR)  # General RAG (inspection, health)
app.include_router(rag_user_files_router, prefix=settings.API_V1_STR)  # User uploads
app.include_router(rag_database_router, prefix=settings.API_V1_STR)  # Database indexes
app.include_router(speech_router, prefix=settings.API_V1_STR)
app.include_router(s3_router, prefix=settings.API_V1_STR)
app.include_router(magic_tools_router, prefix=settings.API_V1_STR)
app.include_router(meeting_summary_router, prefix=settings.API_V1_STR)
app.include_router(legal_tools_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Exab-AI",
        "assistant": "Rashid (راشد) - Saudi Legal AI Assistant",
        "description": "OCR Service for Saudi legal documents and AI-powered legal consultation",
        "providers": {
            "chat": {
                "openai": "GPT models (GPT-3.5, GPT-4, GPT-4o series)",
                "gemini": "Gemini models (Gemini 2.0, 2.5, Flash, Pro, LearnLM)",
                "groq": "Llama models via Groq (fast inference)",
                "total": "Multiple models available"
            },
            "ocr": "Google Gemini 2.0 Flash",
            "web_search": "Perplexity API (optional)"
        },
        "documentation": "/docs",
        "interactive_chat": "/chat.html",
        "s3_browser": "/s3_browser.html",
        "rag_search": "/second_screen.html",
        "pinecone_browser": "/pinecone_browser.html",
        "magic_tools_test": "/magic_tools_test.html",
        "magic_tools_production": "/magic_tools_production.html",
        "legal_tools_test": "/legal_tools_test.html",
        "video_to_audio": "/video_to_audio.html"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/s3_browser.html")
async def s3_browser():
    """Serve S3 browser HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "s3_browser.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "s3_browser.html"
    if not html_path.exists():
        raise FileNotFoundError(f"S3 browser HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/second_screen.html")
async def second_screen():
    """Serve RAG search HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "second_screen.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "second_screen.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Second screen HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/pinecone_browser.html")
async def pinecone_browser():
    """Serve Pinecone browser HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "pinecone_browser.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "pinecone_browser.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Pinecone browser HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/chat.html")
async def chat():
    """Serve chat HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "chat.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "chat.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Chat HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/magic_tools_test.html")
async def magic_tools_test():
    """Serve Magic Tools test HTML page (localhost)"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "magic_tools_test.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "magic_tools_test.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Magic Tools test HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/magic_tools_production.html")
async def magic_tools_production():
    """Serve Magic Tools production HTML page (production server)"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "magic_tools_production.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "magic_tools_production.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Magic Tools production HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/video_to_audio.html")
async def video_to_audio():
    """Serve Video to Audio converter HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "video_to_audio.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "video_to_audio.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Video to Audio HTML file not found at {html_path}")
    return FileResponse(str(html_path))

@app.get("/legal_tools_test.html")
async def legal_tools_test():
    """Serve Legal Tools test HTML page"""
    from pathlib import Path
    # Look in static directory first
    html_path = Path(__file__).parent.parent / "static" / "legal_tools_test.html"
    if not html_path.exists():
        # Fallback to root directory
        html_path = Path(__file__).parent.parent / "legal_tools_test.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Legal Tools test HTML file not found at {html_path}")
    return FileResponse(str(html_path))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    ) 