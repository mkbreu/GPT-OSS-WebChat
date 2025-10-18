# app.py
import streamlit as st
from src.components.sidebar import setup_sidebar
from src.components.chat_window import show_chat

def main():
    st.set_page_config(page_title="GPT‑OSS WebChat", layout="wide")
    setup_sidebar()
    show_chat()

if __name__ == "__main__":
    main()
