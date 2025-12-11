"""Configuration file for Legal RAG Demo"""

import os
from pathlib import Path
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "example_data"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-pro"

# Chunking parameters
MIN_CHUNK_SIZE = 100
DEFAULT_TOP_K = 3

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")