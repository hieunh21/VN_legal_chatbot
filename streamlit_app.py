import json

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Trợ lý Pháp luật VN", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ lý Pháp luật Việt Nam")


def create_session() -> str:
    res = requests.post(f"{API_URL}/sessions")
    return res.json()["id"]


def stream_response(session_id: str, query: str, sources_out: list):
    """
    Sync generator for st.write_stream().
    Yields plain text tokens; populates sources_out as a side effect.
    """
    with requests.post(
        f"{API_URL}/chat/{session_id}/stream",
        json={"query": query},
        stream=True,
    ) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            raw = line[6:]
            if raw == b"[DONE]":
                break
            data = json.loads(raw)
            if data["type"] == "token":
                yield data["content"]
            elif data["type"] == "sources":
                sources_out.extend(data["sources"])


# --- Session state ---
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
                    rel_str = f" 🎯 **{s.get('relevance', 0)}%**" if 'relevance' in s else ""
                    st.markdown(f"- **{s.get('article', '')}** | {s.get('chapter', '')}{rel_str}")


# --- Chat input ---
if query := st.chat_input("Nhập câu hỏi về pháp luật..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    sources: list = []
    with st.chat_message("assistant"):
        try:
            full_text = st.write_stream(
                stream_response(st.session_state.session_id, query, sources)
            )
        except Exception:
            st.error("Lỗi kết nối API. Hãy kiểm tra server FastAPI.")
            full_text = ""

        if sources:
            with st.expander("📚 Nguồn tham khảo"):
                for s in sources:
                    rel_str = f" 🎯 **{s.get('relevance', 0)}%**" if 'relevance' in s else ""
                    st.markdown(f"- **{s.get('article', '')}** | {s.get('chapter', '')}{rel_str}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
    })

