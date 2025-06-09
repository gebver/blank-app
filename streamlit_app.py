import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ========== RAG KOMPONENT ========== #
class SimpleRAG:
    def __init__(self, data_dir="data", embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1"):
        self.texts = []
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self._load_and_embed(data_dir)

    def _load_and_embed(self, data_dir):
        embeddings = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    self.texts.append(text)
                    emb = self.model.encode(text)
                    embeddings.append(emb)
        if embeddings:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))

    def retrieve(self, query, k=2):
        if not self.index:
            return []
        q_vec = self.model.encode([query])
        D, I = self.index.search(np.array(q_vec), k)
        return [self.texts[i] for i in I[0]]

# ========== STREAMLIT APP ========== #
st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app (z RAG)")

# === Pobranie sekretów Streamlit === #
missing = []

api_key = st.secrets.get("API_KEY")
base_url = st.secrets.get("BASE_URL")
selected_model = st.secrets.get("MODEL", "mistralai/mistral-7b-instruct:free")

if not api_key:
    missing.append("API_KEY")
if not base_url:
    missing.append("BASE_URL")

if missing:
    st.error(f"Brakuje następujących sekretów w pliku `.streamlit/secrets.toml`: {', '.join(missing)}")
    st.stop()

# Wczytaj RAG
rag = SimpleRAG(data_dir="data")

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "W czym mogę pomóc? Zadaj pytanie związane z artykułami."}]

# Wyświetl historię
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Obsługa inputu użytkownika
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # RAG – przeszukiwanie dokumentów
    context_docs = rag.retrieve(prompt, k=2)
    context_text = "\n\n".join(context_docs)

    # Stworzenie pełnego prompta
    full_prompt = f"Oto przydatne informacje z dokumentów:\n\n{context_text}\n\nPytanie użytkownika: {prompt}"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem naukowym. Odpowiadaj na podstawie dostarczonego kontekstu."},
            {"role": "user", "content": full_prompt}
        ]
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
