import streamlit as st
import ollama


# ============================================================
# CONFIG
# ============================================================

MODEL = "tinyllama"

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer questions clearly and accurately.
Explain difficult concepts in simple language.
Use examples and step-by-step explanations when useful.
Do not invent information.
If you are unsure, say so.
"""


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Local AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Local AI Chatbot")

st.caption(
    "Powered by Ollama • tinyllama • Running locally"
)


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Type your message..."
)


# ============================================================
# WHEN USER SENDS MESSAGE
# ============================================================

if prompt:

    # Show user message

    with st.chat_message("user"):

        st.markdown(prompt)


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Build conversation

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(
        st.session_state.messages
    )


    # Ask Ollama

    with st.chat_message("assistant"):

        try:

            with st.spinner("Thinking..."):

                response = ollama.chat(
                    model=MODEL,
                    messages=messages
                )


            answer = response["message"]["content"]

            st.markdown(answer)


            # Save AI response

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


        except Exception as e:

            st.error(
                "Could not connect to Ollama."
            )

            st.code(
                str(e)
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("**Model:**")

    st.code(MODEL)

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()