# src/components/sidebar.py
import json
from pathlib import Path
import streamlit as st
from datetime import datetime
from src.utils.ollama_client import OllamaClient
from src.utils.tensorflow_chat import (
    gpu_summary,
    tensorflow_stack_available,
)

# ------- 1. Persistência de conversas ------------------------------------
HIST_DIR = Path(__file__).resolve().parent.parent.parent / "conversations"
HIST_DIR.mkdir(exist_ok=True)

def _save_conversation(convo_id: str, history: list[dict]) -> None:
    (HIST_DIR / f"{convo_id}.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )

def _new_conversation_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ------- 2. Contador de mensagens -----------------------------------------
def _count_user_assistant(history: list[dict]) -> int:
    return sum(1 for m in history if m.get("role") in {"user", "assistant"})

# ------- 3. Sidebar -------------------------------------------------------
def setup_sidebar() -> None:
    st.sidebar.title("📚 Configuração")

    backend_options = ["Ollama (HTTP)"]
    if tensorflow_stack_available():
        backend_options.append("TensorFlow (local)")

    if st.session_state.get("backend") not in backend_options:
        st.session_state["backend"] = backend_options[0]

    backend_choice = st.sidebar.selectbox(
        label="Mecanismo de geração",
        options=backend_options,
        index=backend_options.index(st.session_state.get("backend", backend_options[0])),
        key="backend",
        help="Escolha entre usar o Ollama via API HTTP ou um modelo TensorFlow local.",
    )

    if backend_choice == "Ollama (HTTP)":
        client = OllamaClient()
        try:
            models = client.list_models()
        except Exception as exc:
            st.sidebar.error(f"Falha ao obter modelos do Ollama: {exc}")
            models = ["gpt-oss:20b"]

        default_model = "gpt-oss:20b" if "gpt-oss:20b" in models else models[0]
        model_choice = st.sidebar.selectbox(
            label="Modelo",
            options=models,
            index=models.index(default_model),
            key="ollama_model_choice",
        )
    else:
        st.sidebar.info(gpu_summary())
        default_local_model = st.session_state.get("tf_model_name", "gpt2")
        model_choice = st.sidebar.text_input(
            "Modelo TensorFlow (Hugging Face)",
            value=default_local_model,
            key="tf_model_name",
            help="Nome do modelo publicado no Hugging Face compatível com TensorFlow.",
        )
        st.sidebar.number_input(
            "Tokens máximos gerados",
            min_value=32,
            max_value=1024,
            value=st.session_state.get("tf_max_new_tokens", 256),
            step=32,
            key="tf_max_new_tokens",
            help="Quantidade de tokens adicionais a serem gerados pelo modelo TensorFlow.",
        )

    effort = st.sidebar.selectbox(
        "Tipo de resposta",
        ["conciso", "exploratória", "detalhada"],
        index=0,
        key="effort",
    )

    temperature = st.sidebar.slider(
        "Temperatura",
        min_value=0.3, max_value=2.0, value=1.3, step=0.1,
        key="temperature",
        help="Mais alto → respostas mais criativas",
    )

    show_reasoning = st.sidebar.checkbox(
        "Mostrar raciocínio (chain-of-thought)",
        value=True,
        key="show_reasoning",
    )
    show_metrics = st.sidebar.checkbox(
        "Mostrar métricas de uso",
        value=True,
        key="show_metrics",
    )

    context = st.sidebar.text_area(
        "Contexto adicional (opcional)",
        height=100,
        placeholder="Ex.: Dados históricos, instruções específicas, etc.",
        key="context",
    )

    # ✅ AQUI ESTÁ AJUSTADO O INTERVALO DE TOKENS
    context_size = st.sidebar.slider(
        "Tamanho do contexto (tokens)",
        min_value=4096,
        max_value=256000,
        value=4096,
        step=512,
        key="context_size",
        help="Aumenta a quantidade de tokens que o modelo pode armazenar na conversa",
    )

    st.sidebar.subheader("Anexar arquivos (vários formatos)")
    uploaded_files = st.sidebar.file_uploader(
        "Selecione arquivos (vários)",
        type=[
            "csv", "xlsx", "xls", "ods",
            "txt", "md", "rtf", "docx", "pdf",
            "pptx", "odp",
            "png", "jpg", "jpeg", "gif", "webp",
        ],
        accept_multiple_files=True,
        key="uploaded_files",
        help="Você pode anexar planilhas, PDFs, imagens e documentos para o modelo analisar.",
    )

    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("❌ Limpar conversa", type="primary", key="clear_history"):
        st.session_state["history"] = []

    if col2.button("🆕 Novo chat", type="primary", key="new_chat"):
        convo_id = st.session_state.get("current_convo_id")
        if convo_id and st.session_state.get("history"):
            _save_conversation(convo_id, st.session_state["history"])
        st.session_state["current_convo_id"] = _new_conversation_id()
        st.session_state["history"] = []

    history = st.session_state.get("history", [])
    if _count_user_assistant(history) >= 30:
        st.sidebar.warning(
            "⚠️ 30 mensagens trocadas. Para manter o desempenho, crie um novo chat."
        )

    st.session_state["model_choice"] = model_choice
    st.session_state.setdefault("effort", effort)
    st.session_state.setdefault("temperature", temperature)
    st.session_state.setdefault("show_reasoning", show_reasoning)
    st.session_state.setdefault("show_metrics", show_metrics)
    st.session_state.setdefault("context", context)
    st.session_state.setdefault("context_size", context_size)
    st.session_state.setdefault("uploaded_files", uploaded_files)
    st.session_state.setdefault("backend", backend_choice)
    if backend_choice == "TensorFlow (local)":
        st.session_state.setdefault("tf_model_name", model_choice or "gpt2")
        st.session_state.setdefault("tf_max_new_tokens", 256)
