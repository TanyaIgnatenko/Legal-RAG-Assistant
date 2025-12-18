import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(180deg, #f0f4ff 0%, #ffffff 100%);
    }
    
    /* Header styling */
    .header {
        display: flex;
        align-items: center;
        padding: 1rem 2rem;
        background: white;
        border-bottom: 1px solid #e5e7eb;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .header-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
    }
    
    .header-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.85rem;
        color: #6b7280;
        margin: 0;
    }
    
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        max-width: 600px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        height: 100%;
        border: 1px solid #f0f0f0;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    
    .card-icon-blue {
        background: #e8f0fe;
        color: #3b82f6;
    }
    
    .card-icon-purple {
        background: #f3e8ff;
        color: #8b5cf6;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 50px;
        padding: 1.5rem;
        text-align: center;
        background: #fafafa;
        cursor: pointer;
    }
    
    .upload-text {
        font-size: 0.9rem;
        color: #374151;
        font-weight: 500;
    }
    
    .upload-subtext {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    
    /* Privacy notice */
    .privacy-notice {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        background: #f3f4f6;
        color: #374151;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "document" not in st.session_state:
    st.session_state.document = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def navigate_to_chat(doc_name, chunks=26):
    st.session_state.page = "chat"
    st.session_state.document = {"name": doc_name, "chunks": chunks}
    st.session_state.messages = []

def navigate_to_home():
    st.session_state.page = "home"
    st.session_state.document = None
    st.session_state.messages = []

# Header
col1, col2, col3 = st.columns([1, 10, 1])
with col1:
    if st.session_state.page == "chat":
        if st.button("←", key="back"):
            navigate_to_home()
            st.rerun()

with col2:
    if st.session_state.page == "chat" and st.session_state.document:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                ⚖️
            </div>
            <div>
                <div style="font-weight: 600; color: #1a1a2e;">Legal RAG Assistant</div>
                <div style="font-size: 0.85rem; color: #6b7280;">📄 {st.session_state.document['name']} • {st.session_state.document['chunks']} chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                ⚖️
            </div>
            <div>
                <div style="font-weight: 600; color: #1a1a2e;">Legal RAG Assistant</div>
                <div style="font-size: 0.85rem; color: #6b7280;">AI-powered legal document analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("⚙️", help="Settings")

st.markdown("---")

# Page routing
if st.session_state.page == "home":
    # Home Page
    st.markdown("")
    st.markdown("")
    
    # Main title
    st.markdown("<h1 style='text-align: center; font-size: 2.5rem; font-weight: 700; color: #1a1a2e;'>Understand Legal Documents with AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6b7280; font-size: 1.1rem; max-width: 600px; margin: 0 auto 3rem auto;'>Upload a legal document or use our pre-loaded GDPR file to start asking questions and get AI-powered answers with source citations.</p>", unsafe_allow_html=True)
    
    # Cards
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col2:
        card_col1, card_col2 = st.columns(2, gap="large")
        
        with card_col1:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f0f0f0; height: 300px;">
                <div style="width: 48px; height: 48px; background: #e8f0fe; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
                    📄
                </div>
                <h3 style="font-size: 1.25rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">GDPR Document</h3>
                <p style="font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem; line-height: 1.5;">Pre-loaded General Data Protection Regulation for quick start analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Use GDPR →", key="use_gdpr", use_container_width=True):
                navigate_to_chat("gdpr.pdf", 26)
                st.rerun()
        
        with card_col2:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f0f0f0;">
                <div style="width: 48px; height: 48px; background: #f3e8ff; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
                    📤
                </div>
                <h3 style="font-size: 1.25rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">Upload Document</h3>
                <p style="font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem; line-height: 1.5;">Upload your own PDF or text file to analyze specific legal contracts.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Drag & drop or click to browse",
                type=["pdf", "txt"],
                help="PDF, TXT up to 10MB",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                navigate_to_chat(uploaded_file.name, 15)
                st.rerun()
    
    # Privacy notice
    st.markdown("")
    st.markdown("")
    st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>🔒 Your documents are processed locally and never stored on our servers.</p>", unsafe_allow_html=True)

else:
    # Chat Page
    st.markdown("")
    
    # Suggested questions (only show if no messages yet)
    if not st.session_state.messages:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #e8f0fe 0%, #dbeafe 100%); border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 1rem; font-size: 1.5rem;">
                    🤖
                </div>
                <h2 style="font-size: 1.5rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">Ready to analyze gdpr.pdf</h2>
                <p style="color: #6b7280; font-size: 0.9rem;">✨ Try asking one of these questions:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggested questions
            suggestions = [
                "What are the main principles of data processing under GDPR?",
                "What rights do data subjects have?",
                "What are the penalties for GDPR violations?",
                "When is a Data Protection Officer required?",
                "How long does an organization have to report a data breach?"
            ]
            
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button(suggestions[0], key="q1", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestions[0]})
                    st.rerun()
                if st.button(suggestions[2], key="q3", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestions[2]})
                    st.rerun()
            
            with q_col2:
                if st.button(suggestions[1], key="q2", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestions[1]})
                    st.rerun()
                if st.button(suggestions[3], key="q4", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestions[3]})
                    st.rerun()
            
            if st.button(suggestions[4], key="q5", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestions[4]})
                st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    st.markdown("")
    st.markdown("")
    
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Simulated AI response
        response = f"Based on the {st.session_state.document['name']} document, here's what I found regarding your question:\n\n[This is a placeholder response. In a real implementation, this would be connected to a RAG system that retrieves relevant passages from the document and generates an AI-powered answer with citations.]"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Disclaimer
    st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.75rem; margin-top: 1rem;'>AI responses may vary. Check important information against the source document.</p>", unsafe_allow_html=True)
