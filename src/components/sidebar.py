# src/components/sidebar.py
import json
from pathlib import Path
import streamlit as st
from datetime import datetime
from src.utils.ollama_client import OllamaClient

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
        key="model_choice",
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

    st.session_state.setdefault("model_choice", model_choice)
    st.session_state.setdefault("effort", effort)
    st.session_state.setdefault("temperature", temperature)
    st.session_state.setdefault("show_reasoning", show_reasoning)
    st.session_state.setdefault("show_metrics", show_metrics)
    st.session_state.setdefault("context", context)
    st.session_state.setdefault("context_size", context_size)
    st.session_state.setdefault("uploaded_files", uploaded_files)
