import os
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from groq import Groq

# Load .env from the same directory as this script (works regardless of cwd)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF Chat AI Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
.main-header { text-align: center; padding: 2rem 0 1rem; }
.main-header h1 {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-header p { color: rgba(255,255,255,0.55); font-size: 1rem; }
.chat-container {
    max-height: 520px; overflow-y: auto; padding: 1rem;
    display: flex; flex-direction: column; gap: 1rem;
    scrollbar-width: thin; scrollbar-color: rgba(167,139,250,0.4) transparent;
}
.user-msg {
    align-self: flex-end;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: #fff; padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px; max-width: 75%;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35);
    font-size: 0.95rem; line-height: 1.55;
}
.bot-msg {
    align-self: flex-start;
    background: rgba(255,255,255,0.07); color: #e2e8f0;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px; max-width: 75%;
    border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px);
    font-size: 0.95rem; line-height: 1.55;
}
.msg-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
    margin-bottom: 0.3rem; opacity: 0.6;
}
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(52,211,153,0.15); color: #34d399;
    padding: 0.3rem 0.8rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    border: 1px solid rgba(52,211,153,0.3); margin-bottom: 1rem;
}
.stats-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 1rem 1.2rem;
    margin-bottom: 1rem; color: #e2e8f0;
}
.stats-card h4 {
    margin: 0 0 0.5rem; color: #a78bfa;
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
}
.stats-card p { margin: 0; font-size: 1.4rem; font-weight: 700; color: #fff; }
/* ── Input field fix ── */
[data-baseweb="base-input"],
[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input"] {
    background: #1e1b3a !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
}
[data-baseweb="base-input"] input,
[data-baseweb="input"] input,
.stTextInput input,
[data-testid="stTextInput"] input,
.stForm input[type="text"] {
    background: transparent !important;
    color: #e2e8f0 !important;
    caret-color: #a78bfa !important;
    font-size: 0.95rem !important;
}
[data-baseweb="base-input"] input::placeholder,
[data-baseweb="input"] input::placeholder,
.stTextInput input::placeholder {
    color: rgba(200,200,230,0.4) !important;
}
[data-baseweb="base-input"]:focus-within,
[data-baseweb="input"]:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.35) !important;
    border-radius: 14px !important; padding: 1rem !important;
}
hr { border-color: rgba(255,255,255,0.08) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.4); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files):
    combined_text = ""
    total_pages = 0
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        total_pages += len(reader.pages)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n"
    return combined_text, total_pages


def split_text_into_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_text(raw_text)


def build_bm25_retriever(chunks, k=4):
    """Build a BM25 retriever — pure keyword search, no embeddings, near-instant."""
    docs = [Document(page_content=c) for c in chunks]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def get_relevant_context(retriever, question):
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])


def ask_groq(question, context, chat_history, api_key):
    """Call Groq directly using the official SDK."""
    client = Groq(api_key=api_key)

    # Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant that answers questions based on the provided document context. "
                "Use only the context below to answer. If the answer is not in the context, say so clearly.\n\n"
                f"DOCUMENT CONTEXT:\n{context}"
            ),
        }
    ]

    # Add last 6 turns of history for memory
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def render_chat(history):
    html = '<div class="chat-container">'
    for msg in history:
        if msg["role"] == "user":
            html += f'<div class="user-msg"><div class="msg-label">YOU</div>{msg["content"]}</div>'
        else:
            html += f'<div class="bot-msg"><div class="msg-label">🤖 AI AGENT</div>{msg["content"]}</div>'
    html += "</div>"
    return html


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key, default in [
    ("retriever", None),
    ("chat_history", []),
    ("processed", False),
    ("num_pages", 0),
    ("num_chunks", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    # First check if key is set in environment/dotenv
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key == "your_groq_api_key_here":
        env_key = ""
    
    # If not provided in environment, show input field in the sidebar
    if not env_key:
        st.markdown("## 🔑 API Configuration")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your API key from the Groq Console (https://console.groq.com/)"
        )
        st.markdown("---")
    else:
        groq_api_key = env_key
    
    st.markdown("## 📚 Upload Documents")
    st.markdown("---")

    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed",
    )

    if st.button("⚡ Process Documents", use_container_width=True):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            progress = st.progress(0, text="📄 Extracting text from PDFs...")
            raw_text, total_pages = extract_text_from_pdfs(pdf_docs)
            if not raw_text.strip():
                progress.empty()
                st.error("No text could be extracted from the PDFs.")
            else:
                progress.progress(50, text="✂️ Splitting into chunks...")
                chunks = split_text_into_chunks(raw_text)

                progress.progress(85, text="⚡ Building BM25 index...")
                st.session_state.retriever = build_bm25_retriever(chunks)

                progress.progress(100, text="✅ Done!")
                st.session_state.processed = True
                st.session_state.num_pages = total_pages
                st.session_state.num_chunks = len(chunks)
                st.session_state.chat_history = []
                progress.empty()
                st.success(f"✅ Processed {len(pdf_docs)} PDF(s) · {total_pages} pages · {len(chunks)} chunks")

    st.markdown("---")

    if st.session_state.processed:
        st.markdown('<div class="status-badge">🟢 Ready to Chat</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stats-card"><h4>Pages</h4><p>{st.session_state.num_pages}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stats-card"><h4>Chunks</h4><p>{st.session_state.num_chunks}</p></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload PDFs and click **Process Documents** to start chatting.")

    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;text-align:center;">
        Powered by <strong style="color:#a78bfa;">Groq</strong> +
        <strong style="color:#60a5fa;">LangChain</strong> +
        <strong style="color:#34d399;">BM25</strong>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 Multi-PDF Chat AI Agent</h1>
    <p>Upload your documents and ask questions in natural language</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if st.session_state.chat_history:
    st.markdown(render_chat(st.session_state.chat_history), unsafe_allow_html=True)
else:
    if st.session_state.processed:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:rgba(255,255,255,0.4);">
            <div style="font-size:3rem;margin-bottom:1rem;">💬</div>
            <p>Documents ready! Ask your first question below.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:rgba(255,255,255,0.4);">
            <div style="font-size:4rem;margin-bottom:1rem;">📄</div>
            <p style="font-size:1.1rem;">Upload PDF documents in the sidebar to get started</p>
            <p style="font-size:0.85rem;margin-top:0.5rem;">Supports multi-hop questions across multiple documents</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the key findings in the document?",
            label_visibility="collapsed",
        )
    with col2:
        submit = st.form_submit_button("Send 🚀", use_container_width=True)

if submit and user_question:
    if not groq_api_key or not groq_api_key.strip():
        st.error("🔑 Groq API Key is missing. Please set the GROQ_API_KEY environment variable, add it to your .env file, or enter it in the sidebar.")
    elif not st.session_state.processed:
        st.warning("⚠️ Please upload and process your PDF documents first.")
    else:
        with st.spinner("Thinking..."):
            try:
                context = get_relevant_context(st.session_state.retriever, user_question)
                answer = ask_groq(user_question, context, st.session_state.chat_history, groq_api_key.strip())
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
