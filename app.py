"""
Streamlit Frontend for Legal RAG Demo - Standalone Version
"""

import streamlit as st
import os
from pathlib import Path

from src.rag_system import RAGDemo
from config import get_api_key

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
    [data-testid="stSidebar"] {
        min-width: 286px;
        max-width: 286px;
    }
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
    .example-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        cursor: pointer;
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
    GEMINI_API_KEY = get_api_key()
    
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY not configured!")
        st.info("💡 Add your API key in Streamlit Cloud secrets or set environment variable")
        st.stop()
    
    try:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RAGDemo(gemini_api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing RAG: {str(e)}")
        st.stop()

# Sidebar
with st.sidebar:
    st.title("⚖️ Legal RAG Assistant")
    st.markdown("---")
    
    # Document Source
    st.subheader("📁 Document Source")
    
    doc_source = st.radio(
        "Select source:",
        ["GDPR (Pre-loaded)", "Upload Custom Document"],
        help="Choose between pre-loaded GDPR or upload your own PDF"
    )
    
    if doc_source == "GDPR (Pre-loaded)":
        st.info("📄 Using example_data/gdpr.pdf")
        
        if st.button("Load GDPR Document"):
            if st.session_state.rag_system:
                gdpr_path = "example_data/gdpr.pdf"
                
                if Path(gdpr_path).exists():
                    with st.spinner("Loading GDPR document..."):
                        try:
                            st.session_state.rag_system.setup(gdpr_path)
                            st.session_state.document_loaded = True
                            st.session_state.chunks_count = len(
                                st.session_state.rag_system.vector_store.chunks
                            )
                            st.success("✓ GDPR loaded successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading GDPR: {str(e)}")
                else:
                    st.error(f"❌ File not found: {gdpr_path}")
                    st.info("💡 Place gdpr.pdf in example_data/ directory")
            else:
                st.warning("RAG system not initialized")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a legal document in PDF format"
        )
        
        if uploaded_file and st.button("Process Document"):
            if st.session_state.rag_system:
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process with RAG system
                        st.session_state.rag_system.setup(temp_path)
                        st.session_state.document_loaded = True
                        st.session_state.chunks_count = len(
                            st.session_state.rag_system.vector_store.chunks
                        )
                        
                        # Clean up
                        os.remove(temp_path)
                        st.success(f"✓ {uploaded_file.name} processed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            else:
                st.warning("RAG system not initialized")
    
    # Document Status
    if st.session_state.document_loaded:
        st.markdown("---")
        st.subheader("📊 Document Status")
        st.markdown(f"""
            <div class="status-box">
                <strong>✓ Document Loaded</strong><br>
                <small>{st.session_state.chunks_count} chunks indexed</small>
            </div>
        """, unsafe_allow_html=True)
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many relevant chunks to retrieve for each question"
    )
    
    show_chunks = st.checkbox(
        "Show retrieved chunks",
        value=True,
        help="Display the source chunks used to generate answers"
    )
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Content
st.title("💬 Legal Document Q&A")

# Show welcome message if no document loaded
if not st.session_state.document_loaded:
    st.info("👈 Please load a document from the sidebar to get started!")
    
    st.markdown("### 🎯 How it works:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            **1. Upload Document**
            - Load GDPR or upload your PDF
            - System processes automatically
        """)
    
    with col2:
        st.markdown("""
            **2. Ask Questions**
            - Type your legal questions
            - Get AI-powered answers
        """)
    
    with col3:
        st.markdown("""
            **3. Review Sources**
            - See retrieved document chunks
            - Verify answer accuracy
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Prerequisites:")
    st.code("""
# 1. Install dependencies:
pip install streamlit pymupdf sentence-transformers faiss-cpu google-generativeai numpy

# 2. Set your API key in app.py:
GEMINI_API_KEY = "your-api-key-here"

# 3. Place your PDF (e.g., gdpr.pdf) in the same directory

# 4. Run the app:
streamlit run app.py
    """, language="bash")
    
else:
    # Example Questions
    st.markdown("### 💡 Example Questions")
    
    example_questions = [
        "What is personal data according to GDPR?",
        "What are the rights of data subjects?",
        "What are the penalties for GDPR violations?",
        "What is the legal basis for data processing?",
        "What are the principles of data protection?"
    ]
    
    cols = st.columns(3)
    for idx, question in enumerate(example_questions[:3]):
        with cols[idx]:
            if st.button(question, key=f"example_{idx}"):
                # Add question to chat
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                # Trigger answer generation
                st.rerun()
    
    st.markdown("---")
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation")
        
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
                    # Show retrieved chunks if enabled
                    if show_chunks and "chunks" in message:
                        with st.expander(f"📄 View Retrieved Chunks ({len(message['chunks'])})"):
                            for i, (chunk, distance) in enumerate(message["chunks"], 1):
                                st.markdown(f"""
                                    <div class="chunk-card">
                                        <strong>{i}. {chunk['metadata']}</strong> 
                                        <small>(distance: {distance:.4f})</small>
                                        <p style="margin-top: 0.5rem; color: #64748b;">
                                            {chunk['text'][:300]}...
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
    
    # Question Input
    st.markdown("---")
    
    # Use chat_input for better UX
    if prompt := st.chat_input("Ask your question about the document..."):
        # Add user question to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state.rag_system.answer(
                        prompt, 
                        top_k=top_k, 
                        verbose=False
                    )
                    chunks = st.session_state.rag_system.vector_store.search(
                        prompt, 
                        top_k=top_k
                    )
                    
                    st.markdown(answer)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks
                    })
                    
                    # Show chunks if enabled
                    if show_chunks and chunks:
                        with st.expander(f"📄 View Retrieved Chunks ({len(chunks)})"):
                            for i, (chunk, distance) in enumerate(chunks, 1):
                                st.markdown(f"""
                                    <div class="chunk-card">
                                        <strong>{i}. {chunk['metadata']}</strong> 
                                        <small>(distance: {distance:.4f})</small>
                                        <p style="margin-top: 0.5rem; color: #64748b;">
                                            {chunk['text'][:300]}...
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.info("💡 Check if your document is loaded and API key is valid")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <small>Legal RAG Assistant | Powered by Gemini & FAISS | Built with Streamlit</small>
    </div>
""", unsafe_allow_html=True)