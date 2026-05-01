import streamlit as st
from agent import app, setup_rag
import uuid
import os

# SESSION INIT
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "use_web" not in st.session_state:
    st.session_state.use_web = True

# INIT RAG
@st.cache_resource
def initialize_rag():
    setup_rag()

initialize_rag()

# SIDEBAR
with st.sidebar:
    st.header(" Controls")

    if st.button(" New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

    st.write("---")

    #  Web toggle
    st.subheader(" Web Search")
    st.session_state.use_web = st.toggle(
        "Enable Web Search",
        value=True
    )

    st.write("---")

    st.subheader(" Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)

        for file in uploaded_files:
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.getbuffer())

        setup_rag()
        st.success("Documents added and indexed!")


# MAIN UI
st.title(" Agentic AI Study Assistant")

# DISPLAY CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# USER INPUT
user_input = st.chat_input("Ask something about your documents...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = app.invoke(
                {
                    "question": user_input,
                    "messages": [m["content"] for m in st.session_state.messages],
                    "eval_retries": 0
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                        "use_web": st.session_state.use_web   # pass flag
                    }
                }
            )

            answer = result["answer"]

            #  Show web usage hint
            if st.session_state.use_web and "I don't know" not in answer:
                st.caption("🔎 Used document context or web search if needed")

            st.write(answer)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})
