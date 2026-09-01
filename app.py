"""
Research Paper Assistant
-------------------------
Architecture:  User -> RAG (FAISS retriever) -> LLM (Groq: Llama 3.3 70B)
 
Run with:  streamlit run app.py
Requires:  GROQ_API_KEY set as an environment variable (see .env.example)
"""
 
import os
import streamlit as st
from dotenv import load_dotenv
 
from src.rag import build_index
from src.agent import summarize_paper, answer_question, compare_papers
 
load_dotenv()
 
st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
 
# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "papers" not in st.session_state:
    st.session_state.papers = {}       # name -> PaperIndex
if "summaries" not in st.session_state:
    st.session_state.summaries = {}    # name -> summary text
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {} # name -> list[(role, msg)]
# ---------------------------------------------------------------------------
# Sidebar: API key + uploads
# ---------------------------------------------------------------------------
st.sidebar.title("📄 Research Paper Assistant")
st.sidebar.caption("RAG + LLM assistant for understanding academic papers")
 
if not os.environ.get("GROQ_API_KEY"):
    st.sidebar.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()
 
st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload research paper(s) (PDF)", type=["pdf"], accept_multiple_files=True
)
 
if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.papers:
            with st.spinner(f"Indexing {f.name} ..."):
                index = build_index(f.read(), f.name)
                st.session_state.papers[f.name] = index
                st.session_state.chat_history[f.name] = []
            st.sidebar.success(f"Indexed: {f.name}")
 
paper_names = list(st.session_state.papers.keys())
 
st.sidebar.divider()
st.sidebar.markdown(
    "**Architecture**\n\n"
    "```\nUser\n  ↓\nRAG (FAISS + local embeddings)\n  ↓\nLLM (Groq: Llama 3.3 70B)\n  ↓\nResponse\n```"
)
 
# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Research Paper Assistant")
 
if not paper_names:
    st.info("Upload one or more research paper PDFs from the sidebar to get started.")
    st.stop()
 
tab_summary, tab_qa, tab_compare = st.tabs(
    ["📝 Summary & Analysis", "💬 Ask Questions", "⚖️ Compare Papers"]
)
 
# --- Tab 1: Summary ---------------------------------------------------------
with tab_summary:
    selected = st.selectbox("Select a paper", paper_names, key="summary_select")
    if st.button("Generate Summary", key="gen_summary"):
        with st.spinner("Analyzing paper..."):
            summary = summarize_paper(st.session_state.papers[selected])
            st.session_state.summaries[selected] = summary
 
    if selected in st.session_state.summaries:
        st.markdown(st.session_state.summaries[selected])
    else:
        st.caption("Click 'Generate Summary' to produce Summary, Key Contributions, Limitations, and Future Work.")
 
# --- Tab 2: Q&A --------------------------------------------------------------
with tab_qa:
    selected_qa = st.selectbox("Select a paper", paper_names, key="qa_select")
 
    for role, msg in st.session_state.chat_history[selected_qa]:
        with st.chat_message("user" if role == "User" else "assistant"):
            st.markdown(msg)
 
    question = st.chat_input("Ask a question about this paper...")
    if question:
        st.session_state.chat_history[selected_qa].append(("User", question))
        with st.chat_message("user"):
            st.markdown(question)
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(
                    st.session_state.papers[selected_qa],
                    question,
                    st.session_state.chat_history[selected_qa],
                )
                st.markdown(answer)
        st.session_state.chat_history[selected_qa].append(("Assistant", answer))
 
# --- Tab 3: Compare -----------------------------------------------------------
with tab_compare:
    st.caption("Requires at least 2 papers with generated summaries.")
    chosen = st.multiselect("Select papers to compare", paper_names)
 
    if st.button("Compare Papers", disabled=len(chosen) < 2):
        summaries_needed = {}
        with st.spinner("Preparing summaries..."):
            for name in chosen:
                if name not in st.session_state.summaries:
                    st.session_state.summaries[name] = summarize_paper(st.session_state.papers[name])
                summaries_needed[name] = st.session_state.summaries[name]
 
        with st.spinner("Comparing papers..."):
            comparison = compare_papers(summaries_needed)
            st.markdown(comparison)
 
