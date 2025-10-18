# src/components/sidebar.py
import streamlit as st
from pathlib import Path
import json
import os

# Carrega conversas salvas (arquivo JSON na raiz)
HIST_DIR = Path(__file__).resolve().parent.parent.parent / "conversations"
HIST_DIR.mkdir(exist_ok=True)

def _save_conversation(convo_id, history):
    """Salva o histórico em JSON."""
    (HIST_DIR / f"{convo_id}.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))

def _load_conversation(convo_id):
    """Carrega histórico salvo."""
    file = HIST_DIR / f"{convo_id}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    return []

def _new_conversation_id():
    """Gera um ID único baseado em timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_sidebar():
    """Componente completo da barra lateral."""
    st.sidebar.title("📚 Configuração")

    # ---- 1. Seleção de modelo --------------------------------------------
    st.sidebar.subheader("Modelo")
    model_choice = st.sidebar.selectbox(
        "Escolha o modelo",
        ["gpt-oss:20b", "gpt-oss:120b"],
        index=0,
        help="20B: rápido | 120B: mais inteligente"
    )

    # ---- 2. Estratégia de Raciocínio ---------------------------------------
    st.sidebar.subheader("Raciocínio")
    effort = st.sidebar.selectbox(
        "Esforço de raciocínio",
        ["baixo", "médio", "alto"],
        index=1,
        help="Baixo: resposta curta\nMédio: breve raciocínio\nAlto: chain‑of‑thought completo"
    )

    # ---- 3. Temperatura ----------------------------------------------
    st.sidebar.subheader("Temperatura")
    temperature = st.sidebar.slider(
        "Temperatura",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Mais alto → respostas mais criativas"
    )

    # ---- 4. Exibir raciocínio e métricas ----------------------------------
    show_reasoning = st.sidebar.checkbox(
        "Mostrar raciocínio (chain‑of‑thought)",
        value=True,
        help="Exibe a explicação do modelo antes da resposta"
    )
    show_metrics = st.sidebar.checkbox(
        "Mostrar métricas de desempenho",
        value=True,
        help="Tempo de resposta e modelo usado"
    )

    # ---- 5. Contexto (texto) ---------------------------------------------
    st.sidebar.subheader("Contexto (opcional)")
    context = st.sidebar.text_area(
        "Adicione contexto extra que o modelo deve considerar",
        height=100,
        placeholder="Ex.: Dados históricos, instruções específicas, etc."
    )

    # ---- 6. Upload de arquivos CSV ----------------------------------------
    st.sidebar.subheader("Anexar arquivos CSV")
    uploaded_files = st.sidebar.file_uploader(
        "Selecione arquivos CSV (vários)",
        type="csv",
        accept_multiple_files=True,
        help="Somente arquivos CSV serão lidos e enviados ao modelo"
    )

    # ---- 7. Botões de controle --------------------------------------------
    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("❌ Limpar conversa", type="primary"):
        st.session_state["history"] = []
    if col2.button("🆕 Novo chat", type="primary"):
        # Salva a conversa atual (se houver)
        convo_id = st.session_state.get("current_convo_id")
        if convo_id and st.session_state.get("history"):
            _save_conversation(convo_id, st.session_state["history"])
        # Cria novo ID e limpa tudo
        st.session_state["current_convo_id"] = _new_conversation_id()
        st.session_state["history"] = []

    # ---- 8. Aviso de 30 mensagens ------------------------------------------
    # Conta total de mensagens (perguntas + respostas)
    message_count = st.session_state.get("history", []).count(
        lambda x: x.get("role") in {"user", "assistant"}
    )
    if message_count >= 30:
        st.sidebar.warning(
            "⚠️ 30 mensagens trocadas. Para manter o desempenho, crie um novo chat."
        )

    # ---- 9. Guarda as configurações no session_state ----------------------
    st.session_state.update({
        "model_choice": model_choice,
        "effort": effort,
        "temperature": temperature,
        "show_reasoning": show_reasoning,
        "show_metrics": show_metrics,
        "context":
