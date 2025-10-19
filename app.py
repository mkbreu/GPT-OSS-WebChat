import streamlit as st
from src.components.sidebar import setup_sidebar
from src.components.chat_window import show_chat

def main() -> None:
    st.set_page_config(page_title="Chat GPT‑OSS", page_icon="🤖", layout="wide")
    # Sidebar será montado dentro de setup_sidebar()
    setup_sidebar()
    # Janela principal do chat
    show_chat()

if __name__ == "__main__":
    main()
