# src/components/chat_window.py
# (SUBSTITUA O ARQUIVO INTEIRO POR ESTE)

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
from src.utils.tensorflow_chat import generate_with_tensorflow

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

def _build_prompt(history: list[dict], context_size: int) -> tuple[str, bool]:
    prompt_parts = []

    if st.session_state.get("context"):
        prompt_parts.append(st.session_state["context"])

    # 🔥 ANEXOS SEMPRE ANTES DA PERGUNTA
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

    # 🔹 HISTÓRICO DA CONVERSA AGORA
    for m in history:
        role = "Assistant" if m["role"] == "assistant" else "User"
        prompt_parts.append(f"**{role}**: {m['content']}")

    full_prompt = "\n\n".join(prompt_parts)
    exceeded = False
    token_estimate = len(full_prompt.split())
    if token_estimate > context_size:
        truncated_prompt, exceeded = _truncate_to_fit(full_prompt, context_size, history)
        return truncated_prompt, exceeded
    return full_prompt, False

def generate_ollama_response(prompt: str, model: str, temperature: float = 1.0) -> str:
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

    # ✅ Mostrar anexos antes de enviar
    for f in st.session_state.get("uploaded_files", []):
        st.info(f"📎 Anexo pronto para análise: **{f.name}**")

    with st.form("chat_form", clear_on_submit=False):
        user_query = st.text_area(
            "Digite sua pergunta",
            key="user_query",
            height=120,
            placeholder="Use Enter ou Ctrl+Enter para enviar ✨",
            label_visibility="collapsed"
        )

        sent = st.form_submit_button("🟢 Enviar")

    if sent:
        if not user_query or not user_query.strip():
            st.warning("⚠️ Pergunta vazia – tente novamente.")
            return

        _append_message(history, "user", user_query)

        prompt, exceeded = _build_prompt(history, st.session_state.get("context_size", 4096))
        if exceeded:
            st.warning("⚠️ Contexto muito grande. Reduza a conversa ou aumente o limite.")

        with st.spinner("Gerando resposta..."):
            backend = st.session_state.get("backend", "Ollama (HTTP)")
            temperature = st.session_state.get("temperature", 1.3)
            if backend == "TensorFlow (local)":
                model_name = st.session_state.get("tf_model_name") or "gpt2"
                max_new_tokens = int(st.session_state.get("tf_max_new_tokens", 256))
                if not model_name.strip():
                    answer = "*Defina um modelo TensorFlow válido na barra lateral.*"
                else:
                    answer = generate_with_tensorflow(
                        prompt,
                        model_name=model_name.strip(),
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
            else:
                answer = generate_ollama_response(
                    prompt,
                    st.session_state.get("model_choice"),
                    temperature,
                )

        _append_message(history, "assistant", answer)

    st.markdown("---")
    st.markdown("## Histórico do Chat")

    # ✅ RENDERIZAÇÃO PERSONALIZADA (MÍNIMO NECESSÁRIO ALTERADO)
    for m in history:
        is_user = m["role"] == "user"
        bg = "#D0E7FF" if is_user else "#F1F1F1"
        align = "right" if is_user else "left"
        st.markdown(
            f"""
            <div style="
                background-color:{bg};
                padding:8px 12px;
                margin:4px 0;
                border-radius:8px;
                text-align:{align};
            ">{m['content']}</div>
            """,
            unsafe_allow_html=True
        )
