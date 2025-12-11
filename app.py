"""
Streamlit Frontend for Legal RAG Demo

Run: streamlit run app.py
"""

import streamlit as st
import os
from pathlib import Path

from src import RAGDemo
from config import GEMINI_API_KEY

# Page configuration
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1e40af;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    .chunk-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #3b82f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'chunks_count' not in st.session_state:
    st.session_state.chunks_count = 0

# Initialize RAG system on startup
if st.session_state.rag_system is None:
    try:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RAGDemo(gemini_api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing RAG: {str(e)}")

# Sidebar
with st.sidebar:
    st.title("⚖️ Legal RAG Assistant")
    st.markdown("---")
    
    st.subheader("📁 Document Source")
    
    doc_source = st.radio(
        "Select source:",
        ["GDPR (Pre-loaded)", "Upload Custom Document"]
    )
    
    if doc_source == "GDPR (Pre-loaded)":
        gdpr_path = st.text_input("GDPR PDF Path:", value="gdpr.pdf")
        
        if st.button("Load GDPR Document"):
            if st.session_state.rag_system and Path(gdpr_path).exists():
                with st.spinner("Loading GDPR..."):
                    try:
                        st.session_state.rag_system.setup(gdpr_path)
                        st.session_state.document_loaded = True
                        st.session_state.chunks_count = len(
                            st.session_state.rag_system.vector_store.chunks
                        )
                        st.success("✓ GDPR loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        
        if uploaded_file and st.button("Process Document"):
            with st.spinner("Processing..."):
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.rag_system.setup(temp_path)
                    st.session_state.document_loaded = True
                    st.session_state.chunks_count = len(
                        st.session_state.rag_system.vector_store.chunks
                    )
                    
                    os.remove(temp_path)
                    st.success(f"✓ {uploaded_file.name} processed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.document_loaded:
        st.markdown("---")
        st.markdown(f"""
            <div class="status-box">
                <strong>✓ Document Loaded</strong><br>
                <small>{st.session_state.chunks_count} chunks indexed</small>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    top_k = st.slider("Number of chunks", 1, 10, 3)
    show_chunks = st.checkbox("Show retrieved chunks", True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main Content
st.title("💬 Legal Document Q&A")

if not st.session_state.document_loaded:
    st.info("👈 Please load a document from the sidebar!")
else:
    # Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and show_chunks and "chunks" in message:
                with st.expander(f"📄 Retrieved Chunks ({len(message['chunks'])})"):
                    for i, (chunk, dist) in enumerate(message["chunks"], 1):
                        st.markdown(f"""
                            <div class="chunk-card">
                                <strong>{i}. {chunk['metadata']}</strong> 
                                <small>(distance: {dist:.4f})</small>
                                <p style="margin-top: 0.5rem;">
                                    {chunk['text'][:300]}...
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Input
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_system.answer(
                        prompt, top_k=top_k, verbose=False
                    )
                    chunks = st.session_state.rag_system.vector_store.search(
                        prompt, top_k=top_k
                    )
                    
                    st.markdown(answer)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b;">
        <small>Legal RAG Assistant | Powered by Gemini & FAISS</small>
    </div>
""", unsafe_allow_html=True)
