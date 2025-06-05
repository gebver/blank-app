import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

# Próba pobrania sekretów ze Streamlit Cloud
try:
    api_key = st.secrets["API_KEY"]
    base_url = st.secrets["BASE_URL"]
except Exception:
    # Jeśli nie ma w st.secrets, bierzemy z environment variables (np. Codespaces)
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")

#selected_model = "mistralai/mistral-7b-instruct:free"
selected_model = os.environ.get("MODEL")


if not api_key or not base_url:
    st.error("Brak API_KEY lub BASE_URL. Ustaw je w sekretach Streamlit lub jako zmienne środowiskowe.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

# Wyświetlanie historii czatu
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Obsługa nowego inputu użytkownika
if prompt := st.chat_input():
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
