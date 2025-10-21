# src/components/sidebar.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

from src.utils.ollama_client import OllamaClient
from src.utils.knowledge_base import build_kb_from_uploads
from src.utils.history_manager import export_history_to_txt, export_history_to_docx

# Persistência de conversas
HIST_DIR = Path(__file__).resolve().parent.parent.parent / "conversations"
HIST_DIR.mkdir(exist_ok=True)
EXPORT_DIR = HIST_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

def _save_conversation(convo_id: str, history: list[dict]) -> None:
    (HIST_DIR / f"{convo_id}.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def _new_conversation_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _count_user_assistant(history: list[dict]) -> int:
    return sum(1 for m in history if m.get("role") in {"user", "assistant"})

def _auto_export_history(history: list[dict]) -> tuple[Path, Path] | None:
    if not history:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = EXPORT_DIR / f"chat_{ts}.txt"
    docx_path = EXPORT_DIR / f"chat_{ts}.docx"
    # TXT
    txt_buf = export_history_to_txt(history)
    txt_path.write_bytes(txt_buf.getvalue())
    # DOCX
    docx_buf = export_history_to_docx(history)
    docx_path.write_bytes(docx_buf.getvalue())
    return txt_path, docx_path

def setup_sidebar() -> None:
    st.sidebar.title("📚 Configuração")

    # Cliente de modelos (Ollama)
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

    context_size = st.sidebar.slider(
        "Tamanho do contexto (tokens)",
        min_value=4096,
        max_value=256000,
        value=4096,
        step=512,
        key="context_size",
        help="Aumenta a quantidade de tokens que o modelo pode considerar na conversa",
    )

    st.sidebar.subheader("Anexar arquivos (vários formatos)")
    uploaded_files = st.sidebar.file_uploader(
        "Selecione arquivos (vários)",
        type=[
            "csv", "xlsx", "xls", "ods",
            "txt", "md", "rtf", "docx", "pdf",
            "pptx", "odp",
            "png", "jpg", "jpeg", "gif", "webp", "tif",
        ],
        accept_multiple_files=True,
        key="uploaded_files",
        help="Planilhas, PDFs, imagens e documentos.",
    )

    # Indexação leve (RAG-lite) assim que houver anexos
    if uploaded_files:
        try:
            kb = build_kb_from_uploads(uploaded_files)
            st.session_state["kb"] = kb
            if kb.use_embeddings:
                st.sidebar.success(f"Base preparada ({len(kb.chunks)} trechos). Embeddings: ok.")
            else:
                st.sidebar.warning(f"Base preparada ({len(kb.chunks)} trechos). Embeddings indisponíveis (fallback: palavras-chave).")
        except Exception as e:
            st.sidebar.error(f"Falha ao preparar base de conhecimento: {e}")
            st.session_state["kb"] = None
    else:
        st.session_state["kb"] = st.session_state.get("kb", None)

    # Botões de controle
    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("❌ Limpar conversa", type="primary", key="clear_history"):
        st.session_state["history"] = []

    # Fluxo de Novo Chat com confirmação e exportação automática
    pending_key = "pending_new_chat"
    if pending_key not in st.session_state:
        st.session_state[pending_key] = False

    if not st.session_state[pending_key]:
        if col2.button("🆕 Novo chat", type="primary", key="new_chat"):
            st.session_state[pending_key] = True
    else:
        with st.sidebar.expander("⚠️ Confirmar novo chat", expanded=True):
            export_auto = st.checkbox("Exportar histórico automaticamente (TXT + DOCX)", value=True, key="export_auto")
            purge_side = st.checkbox("Apagar anexos e contexto da sidebar", value=True, key="purge_side")

            c1, c2 = st.columns(2)
            if c1.button("Confirmar", key="confirm_new_chat"):
                # Exporta antes de limpar
                history = st.session_state.get("history", [])
                if export_auto and history:
                    try:
                        out = _auto_export_history(history)
                        if out:
                            txtp, docxp = out
                            st.sidebar.success(f"💾 Histórico exportado em:\n- {txtp}\n- {docxp}")
                    except Exception as e:
                        st.sidebar.error(f"Falha na exportação automática: {e}")

                # Salva conversa atual (json) se houver
                convo_id = st.session_state.get("current_convo_id")
                if convo_id and st.session_state.get("history"):
                    _save_conversation(convo_id, st.session_state["history"])

                # Reset total de caches/estado
                st.session_state["current_convo_id"] = _new_conversation_id()
                st.session_state["history"] = []
                st.session_state["kb"] = None

                if purge_side:
                    st.session_state["context"] = ""
                    st.session_state["uploaded_files"] = None

                st.session_state[pending_key] = False
                st.sidebar.success("Novo chat iniciado com sucesso.")
            if c2.button("Cancelar", key="cancel_new_chat"):
                st.session_state[pending_key] = False

    # Aviso de 30 mensagens
    history = st.session_state.get("history", [])
    if _count_user_assistant(history) >= 30:
        st.sidebar.warning("⚠️ 30 mensagens trocadas. Para manter o desempenho, crie um novo chat.")

    # Guarda as configurações no session_state (só para garantir)
    st.session_state.setdefault("model_choice", model_choice)
    st.session_state.setdefault("effort", effort)
    st.session_state.setdefault("temperature", temperature)
    st.session_state.setdefault("show_reasoning", show_reasoning)
    st.session_state.setdefault("show_metrics", show_metrics)
    st.session_state.setdefault("context", context)
    st.session_state.setdefault("context_size", context_size)
    st.session_state.setdefault("uploaded_files", uploaded_files)
