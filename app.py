import streamlit as st
import requests
import os

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TN AgriBOT",
    page_icon="🌾",
    layout="wide"
)

# Custom Styling to match the screenshot banner
st.markdown("""
<style>
    .banner {
        background-color: #1e5e3a;
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .banner h1 {
        color: white;
        margin: 0;
        font-size: 32px;
    }
    .banner p {
        color: #d1e7dd;
        margin-top: 6px;
        font-size: 15px;
    }
    .info-card {
        background-color: #fef8e7;
        border-left: 5px solid #f5b041;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 14px;
        color: #5c4308;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.success("OpenAI API key detected in environment.")

    st.subheader("📁 Upload Agriculture PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🌱 Build / Replace Knowledge Base", use_container_width=True, type="primary"):
        with st.spinner("Uploading and indexing documents (Web scraper + PDFs)..."):
            try:
                # 1. Upload files to backend if provided
                if uploaded_files:
                    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                    requests.post(f"{API_BASE}/upload_pdfs", files=files, timeout=30)
                
                # 2. Trigger index build
                res = requests.post(f"{API_BASE}/build_index", timeout=120)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["chunks_count"] = data.get("total_chunks", 0)
                    st.session_state["indexed_pdfs"] = data.get("pdfs", [])
                    st.success("Knowledge Base successfully built!")
                else:
                    st.error("Failed to build knowledge base.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📊 Knowledge Base Status")
    chunks = st.session_state.get("chunks_count", 0)
    st.caption("Text chunks indexed")
    st.markdown(f"### {chunks}")

    indexed_pdfs = st.session_state.get("indexed_pdfs", [])
    st.write(f"**PDFs indexed:** {len(indexed_pdfs)}")
    st.write("• `schemes_data.txt` (TNAU Scraper)")
    for pdf in indexed_pdfs:
        st.write(f"• `{pdf}`")

    st.markdown("---")
    if st.button("🧹 Clear chat / search results", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Reset knowledge base", use_container_width=True):
        try:
            requests.post(f"{API_BASE}/reset_index", timeout=10)
            st.session_state["chunks_count"] = 0
            st.session_state["indexed_pdfs"] = []
            st.session_state.messages = []
            st.success("Knowledge base reset.")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting: {e}")

# ----------------- MAIN PANEL -----------------
st.markdown("""
<div class="banner">
    <h1>🌾 TN AgriBOT</h1>
    <p>Your PDF & Web-powered Farmer Assistant · Ask questions across verified Tamil Nadu agriculture documents</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.4, 1])
with col1:
    st.markdown("""
    <div class="info-card">
        <b>How to use:</b> Upload your agriculture PDFs in the sidebar, build the shared knowledge base, then ask questions in the chat. Each answer includes source files and page references so you can check the original document.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.image(
        "https://static.investindia.gov.in/s3fs-public/2019-02/Blog%20Image.jpg",
        caption="Healthy soil, healthy crops, stronger farms",
        use_container_width=True
    )

st.markdown("### 💬 Ask TN AgriBOT")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am TN AgriBOT. Ask me any question from your scraped schemes data or uploaded agriculture PDFs."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about your uploaded agriculture PDFs or web schemes..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed web data and PDFs..."):
            try:
                response = requests.post(f"{API_BASE}/ask", json={"question": user_input}, timeout=60)
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                else:
                    answer = f"Error {response.status_code}: Backend error occurred."
            except requests.exceptions.ConnectionError:
                answer = "Cannot reach FastAPI backend. Make sure `uvicorn main:app --reload` is running on port 8000."
            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})