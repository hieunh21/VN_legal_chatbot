import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Trợ lý Pháp luật VN", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ lý Pháp luật Việt Nam")


# --- Session management ---
def create_session():
    res = requests.post(f"{API_URL}/sessions")
    return res.json()["id"]


if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.header("Phiên trò chuyện")
    st.caption(f"Session: `{st.session_state.session_id}`")

    if st.button("🔄 Phiên mới"):
        st.session_state.session_id = create_session()
        st.session_state.messages = []
        st.rerun()


# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Nguồn tham khảo"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s.get('article', '')}** | {s.get('chapter', '')}")


# --- Chat input ---
if query := st.chat_input("Nhập câu hỏi về pháp luật..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            res = requests.post(
                f"{API_URL}/chat/{st.session_state.session_id}",
                json={"query": query},
            )

        if res.status_code == 200:
            data = res.json()
            st.markdown(data["answer"])

            if data.get("sources"):
                with st.expander("📚 Nguồn tham khảo"):
                    for s in data["sources"]:
                        st.markdown(f"- **{s.get('article', '')}** | {s.get('chapter', '')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
            })
        else:
            st.error("Lỗi kết nối API. Hãy kiểm tra server FastAPI.")
