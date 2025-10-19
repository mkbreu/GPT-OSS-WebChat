# src/components/chat_window.py
import json
from pathlib import Path
import requests
import streamlit as st
from src.utils.file_reader import (
    read_csv_file,
    read_excel_file,
    read_txt_file,
    read_docx_file,
    read_pdf_file,
    read_pptx_file,
    read_image_file,
)

# ----------------------------------------------------------------------
# 1. Auxiliares de histórico
# ----------------------------------------------------------------------
def _append_message(history, role, content):
    history.append({"role": role, "content": content})

def _truncate_to_fit(prompt: str, max_tokens: int, history: list[dict]) -> tuple[str, bool]:
    tokens = prompt.split()
    if len(tokens) <= max_tokens:
        return prompt, False
    trunc_tokens = len(" ".join(m["content"] for m in history[-5:]).split())
    if trunc_tokens > max_tokens:
        return prompt, True
    new_prompt = " ".join(m["content"] for m in history[-5:]).replace("\n", " ")
    return new_prompt, False

# ----------------------------------------------------------------------
# 2. Construindo o prompt completo
# ----------------------------------------------------------------------
def _build_prompt(history: list[dict], context_size: int) -> tuple[str, bool]:
    prompt_parts = []
    if st.session_state.get("context"):
        prompt_parts.append(st.session_state["context"])
    for m in history:
        role = "Assistant" if m["role"] == "assistant" else "User"
        prompt_parts.append(f"**{role}**: {m['content']}")

    for f in st.session_state.get("uploaded_files", []):
        ext = Path(f.name).suffix.lower()
        try:
            if ext == ".csv":
                text = read_csv_file(f.getvalue())
            elif ext in {".xlsx", ".xls", ".ods"}:
                text = read_excel_file(f.getvalue())
            elif ext in {".txt", ".md", ".rtf"}:
                text = read_txt_file(f.getvalue())
            elif ext == ".docx":
                text = read_docx_file(f.getvalue())
            elif ext == ".pdf":
                text = read_pdf_file(f.getvalue())
            elif ext in {".pptx", ".odp"}:
                text = read_pptx_file(f.getvalue())
            elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif"}:
                text = read_image_file(f.getvalue())
            else:
                text = f"*Tipo de arquivo `{ext}` não suportado*"
        except Exception as e:
            text = f"*Erro lendo `{f.name}`: {e}*"
        prompt_parts.append(f"--- **Arquivo**: {f.name} ---\n{text}\n")

    if "user_query" in st.session_state:
        prompt_parts.append(f"**Pergunta**: {st.session_state['user_query']}")

    full_prompt = "\n\n".join(prompt_parts)
    exceeded = False
    token_estimate = len(full_prompt.split())
    if token_estimate > context_size:
        truncated_prompt, exceeded = _truncate_to_fit(full_prompt, context_size, history)
        return truncated_prompt, exceeded
    return full_prompt, False

# ----------------------------------------------------------------------
# 3. Função de geração de resposta (substitui OllamaClient)
# ----------------------------------------------------------------------
def generate_response(prompt: str, model: str, temperature: float = 1.0) -> str:
    """
    Envia o prompt ao servidor Ollama local e processa o stream de tokens.
    """
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
            # Concatena os tokens de texto progressivamente
            if "response" in data and data["response"]:
                result += data["response"]
            if data.get("done"):
                break
    except Exception as e:
        return f"*Erro ao processar resposta: {e}*"

    if not result.strip():
        return "*Resposta vazia do modelo (verifique se o modelo está totalmente carregado).*"
    return result.strip()

# ----------------------------------------------------------------------
# 4. Janela principal do chat
# ----------------------------------------------------------------------
def show_chat() -> None:
    # Inicializa histórico se necessário
    if "history" not in st.session_state:
        st.session_state["history"] = []

    history = st.session_state["history"]

    st.markdown("## Sua pergunta")

    temperature = st.session_state.get("temperature", 1.3)
    context_size = st.session_state.get("context_size", 4096)
    model_choice = st.session_state.get("model_choice", "gpt-oss:20b")

    user_query = st.text_area(
        label="Digite sua pergunta",
        key="user_query",
        height=120,
        placeholder="Digite sua pergunta e clique em Enviar",
        label_visibility="collapsed",
    )

    if st.button("🟢 Enviar", key="send_msg"):
        if not user_query or not user_query.strip():
            st.warning("⚠️ Pergunta vazia – tente novamente.")
            return

        _append_message(history, "user", user_query)

        prompt, exceeded = _build_prompt(history, context_size)
        if exceeded:
            st.warning(
                f"⚠️ O prompt contém mais de {context_size} tokens. "
                "Reduza o tamanho da conversa ou aumente o contexto na barra lateral."
            )

        with st.spinner("Gerando resposta..."):
            answer = generate_response(prompt, model_choice, temperature)

        _append_message(history, "assistant", answer)

    st.markdown("---")
    st.markdown("## Histórico do Chat")
    for i, m in enumerate(history):
        role = m["role"]
        content = m["content"]
        if role == "assistant":
            st.markdown(f"**Assistant**:\n\n{content}")
        else:
            st.markdown(f"**Você**:\n\n{content}")

    if st.session_state.get("show_metrics"):
        total_tokens = len("\n".join(m["content"] for m in history).split())
        st.sidebar.metric(label="Total de tokens usados", value=total_tokens)
