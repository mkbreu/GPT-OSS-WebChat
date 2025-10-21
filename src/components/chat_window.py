# src/components/chat_window.py
from __future__ import annotations
import json
from pathlib import Path
import requests
import streamlit as st

from src.utils.history_manager import (
    chunk_history_dynamic, export_history_to_txt, export_history_to_docx
)
from src.utils.knowledge_base import retrieve

def _append_message(history, role, content):
    history.append({"role": role, "content": content})

def _estimate_tokens(s: str) -> int:
    # Estimativa simples: 1 token ~= 1 palavra (aproximação)
    return len(s.split())

def _build_prompt(history: list[dict], context_size: int) -> tuple[str, bool, int]:
    """
    PROMPT FINAL = [Contexto digitado] + [Contexto recuperado de anexos]
                 + [Histórico completo (chunking)] + [Pergunta atual]
    NÃO CORTA conteúdo para caber. Se exceder o context_size, retornará exceeded=True.
    """
    parts = []

    # 1) Contexto manual (sidebar)
    if st.session_state.get("context"):
        parts.append(st.session_state["context"])

    # 2) Contexto recuperado (KB)
    kb = st.session_state.get("kb")
    last_user_msg = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "").strip()
    if kb and last_user_msg:
        try:
            retrieved, _ = retrieve(last_user_msg, kb)
            if retrieved:
                parts.append("### Contexto recuperado de anexos\n" + retrieved)
        except Exception:
            pass

    # 3) Histórico completo com chunking
    hist_text, total_chars, n_chunks, _ = chunk_history_dynamic(history, max_chars_per_chunk=8000, overlap=800)
    if hist_text:
        parts.append(hist_text)

    # 4) Pergunta atual já foi incluída no histórico (última mensagem do user)

    full_prompt = "\n\n".join(parts)
    token_estimate = _estimate_tokens(full_prompt)
    exceeded = token_estimate > int(context_size)
    return full_prompt, exceeded, token_estimate

def generate_response(prompt: str, model: str, temperature: float = 1.0) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}}
    try:
        response = requests.post(url, json=payload, stream=True, timeout=600)
    except Exception as e:
        return f"*Erro ao conectar com Ollama: {e}*"

    result = ""
    try:
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data and data["response"]:
                result += data["response"]
            if data.get("done"):
                break
    except Exception as e:
        return f"*Erro ao processar resposta: {e}*"
    return result.strip() or "*Resposta vazia do modelo.*"

def show_chat() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    history = st.session_state["history"]

    st.markdown("## Sua pergunta")

    # Anexos prontos
    for f in st.session_state.get("uploaded_files", []) or []:
        st.info(f"📎 Anexo pronto: **{f.name}**")

    # Export manual (além da exportação automática no novo chat)
    if history:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "⬇️ Exportar histórico (.txt)",
                export_history_to_txt(history),
                file_name="chat_history.txt",
            )
        with col2:
            st.download_button(
                "⬇️ Exportar histórico (.docx)",
                export_history_to_docx(history),
                file_name="chat_history.docx",
            )

    # Campo de pergunta
    user_query = st.text_area(
        "Digite sua pergunta",
        key="user_query",
        height=120,
        placeholder="Digite sua pergunta e pressione Enviar",
        label_visibility="collapsed"
    )

    if st.button("🟢 Enviar", key="send_msg"):
        if not user_query or not user_query.strip():
            st.warning("⚠️ Pergunta vazia – tente novamente.")
            return

        _append_message(history, "user", user_query)

        prompt, exceeded, token_estimate = _build_prompt(
            history, st.session_state.get("context_size", 4096)
        )
        if exceeded:
            st.error(
                f"🚨 O prompt montado excede o limite de contexto configurado "
                f"({token_estimate} > {st.session_state.get('context_size', 4096)} tokens). "
                f"Aumente o **Tamanho do contexto (tokens)** na barra lateral e reenvie. "
                f"Nenhum conteúdo foi cortado para preservar a precisão."
            )
            # Reverte a mensagem do usuário para não poluir o histórico enquanto ajusta o slider
            history.pop()
            return

        with st.spinner("Gerando resposta..."):
            answer = generate_response(
                prompt,
                st.session_state.get("model_choice", "gpt-oss:20b"),
                st.session_state.get("temperature", 1.3),
            )

        _append_message(history, "assistant", answer)
        # Limpa campo após envio
        st.session_state["user_query"] = ""

    # Histórico visual
    st.markdown("---")
    st.markdown("## Histórico do Chat")
    for m in history:
        role = "Assistant" if m["role"] == "assistant" else "Você"
        bg = "#f0f0f0" if role == "Assistant" else "#e0f3ff"
        st.markdown(
            f"""
            <div style='background-color:{bg}; padding:8px 12px; border-radius:10px; margin-bottom:4px;'>
                <b>{role}:</b><br>{m['content']}
            </div>
            """,
            unsafe_allow_html=True
        )
