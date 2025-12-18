import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "example_data"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-pro"

# Chunking parameters
MIN_CHUNK_SIZE = 100
DEFAULT_TOP_K = 3


# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative frontend
    "https://your-frontend-domain.vercel.app",  # Production frontend
]

# File upload settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = [".pdf", ".txt"]

# API Keys - only use environment variables for backend
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")