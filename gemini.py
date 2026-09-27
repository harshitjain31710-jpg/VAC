import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

API_KEY = "YOUR_API_KEY_HERE" # Put your key back here (but don't commit it!)
MODEL = "gemini-3.5-flash"
client = genai.Client(api_key=API_KEY)

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
    page_title="Gemini AI Chatbot",
    page_icon="✨",
    layout="centered"
)

# ============================================================
# TITLE
# ============================================================

st.title("✨ Gemini AI Chatbot")
st.caption(f"Powered by Google Gemini • {MODEL} • Cloud")

# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:
    # We map "model" back to "assistant" for Streamlit's built-in icon
    ui_role = "assistant" if message["role"] == "model" else message["role"]
    with st.chat_message(ui_role):
        st.markdown(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Type your message...")

# ============================================================
# WHEN USER SENDS MESSAGE
# ============================================================

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build conversation for Gemini
    formatted_contents = []
    for msg in st.session_state.messages:
        formatted_contents.append(
            types.Content(
                role=msg["role"], 
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT
    )

    # Ask Gemini
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model=MODEL,
                    contents=formatted_contents,
                    config=config
                )
            
            answer = response.text
            st.markdown(answer)

            # Save AI response
            st.session_state.messages.append({"role": "model", "content": answer})

        except Exception as e:
            st.error("Could not connect to Gemini API.")
            st.code(str(e))

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("**Model:**")
    st.code(MODEL)
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()