import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from agent import build_agent
from ingest import index_pdf

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Study Assistant Agent")
st.caption("Agentic RAG for answering questions from your study PDFs.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload notes")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        if st.button("Index PDF", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.getbuffer())
                pdf_path = f.name
            with st.spinner("Reading and indexing..."):
                st.session_state.vectorstore = index_pdf(pdf_path)
                st.session_state.agent = build_agent(st.session_state.vectorstore)
            st.success("PDF indexed successfully.")

    st.divider()
    st.write("**Agent flow**")
    st.write("Question → Retrieve → Check relevance → Broaden query if needed → Answer")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Add OPENAI_API_KEY to your .env file before asking questions.")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask something from your notes...")

if question:
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.agent is None:
        answer = "Please upload and index a PDF first."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking and retrieving..."):
                result = st.session_state.agent.invoke({"question": question})
                answer = result["answer"]
            st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))
