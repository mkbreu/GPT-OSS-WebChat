# app.py
import os
import subprocess
from datetime import datetime

import streamlit as st
import tensorflow as tf
from tensorflow.python.client import device_lib

# ======= Página =======
st.set_page_config(page_title="GPT-OSS WebChat", page_icon="💬", layout="wide")
st.title("💬 GPT-OSS WebChat")
st.write("Interface interativa baseada em Streamlit com suporte a GPU via TensorFlow")

# ======= Sidebar completa (modelos, uploads, etc.) =======
try:
    from src.components.sidebar import setup_sidebar
    setup_sidebar()
except Exception as e:
    st.sidebar.error("Falha ao carregar a sidebar completa (src/components/sidebar.py).")
    st.sidebar.code(str(e))

# Bloco “sistema” compatível (sem st.divider)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Sistema")

c1, c2 = st.sidebar.columns(2)
if c1.button("🔍 Verificar GPU"):
    try:
        subprocess.Popen(["streamlit", "run", "gpu_status.py"])
        st.sidebar.info("Abrindo painel detalhado de GPU em nova janela...")
    except Exception as e:
        st.sidebar.error(f"Erro ao abrir gpu_status.py: {e}")

if c2.button("🧠 Reiniciar app"):
    os.system("streamlit run app.py")

if st.sidebar.button("🧹 Limpar conversa"):
    st.session_state["messages"] = []
    st.session_state["history"] = []
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 Ambiente Ativo")
st.sidebar.write("**TensorFlow:**", tf.__version__)
st.sidebar.write("**Dispositivos TensorFlow:**", len(tf.config.list_physical_devices()))

# ======= GPU info =======
def show_gpu_info():
    try:
        devices = device_lib.list_local_devices()
        gpus = [d for d in devices if d.device_type == "GPU"]
        if gpus:
            gpu = gpus[0]
            st.success("✅ GPU detectada e ativa!")
            name = gpu.physical_device_desc.split("name: ")[1].split(",")[0]
            st.write(f"**Nome:** {name}")
            st.write(f"**Tipo:** {gpu.device_type}")
            st.write(f"**Memória disponível:** {round(gpu.memory_limit / (1024**3), 2)} GB")

            build = tf.sysconfig.get_build_info()
            st.write(f"**CUDA:** {build.get('cuda_version', 'Desconhecida')}")
            st.write(f"**cuDNN:** {build.get('cudnn_version', 'Desconhecida')}")
        else:
            st.warning("⚠️ Nenhuma GPU detectada. TensorFlow está usando apenas CPU.")
    except Exception as e:
        st.error(f"Erro ao verificar status da GPU: {e}")

# ======= Estado inicial =======
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "user", "content": "Olá! 👋 Este é um teste de ambiente com GPU ativada.", "ts": datetime.now().strftime("%H:%M:%S")},
        {"role": "assistant", "content": "Tudo certo, Maykon. O TensorFlow está operando com aceleração de hardware!", "ts": datetime.now().strftime("%H:%M:%S")},
    ]

# Histórico “canônico” para export/contagem da sidebar
if "history" not in st.session_state:
    st.session_state["history"] = st.session_state["messages"][:]

# ======= Estilo dos balões =======
st.subheader("💬 Chat de Demonstração")
st.markdown("""
<style>
.user-bubble { background-color:#1E88E5;color:#fff;padding:12px;border-radius:12px;margin-bottom:8px;max-width:80%; }
.assistant-bubble { background-color:#E8EAF6;color:#000;padding:12px;border-radius:12px;margin-bottom:8px;max-width:80%; }
.timestamp { font-size:.75em;color:gray;text-align:right;margin-bottom:15px; }
</style>
""", unsafe_allow_html=True)

# ======= Render das mensagens =======
for msg in st.session_state["messages"]:
    bubble = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    prefix = "🧑 Você:" if msg["role"] == "user" else "🤖 Assistente:"
    st.markdown(f"<div class='{bubble}'><b>{prefix}</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='timestamp'>{msg.get('ts','')}</div>", unsafe_allow_html=True)

# ======= Montagem de prompt (Contexto + KB + Histórico) =======
from src.utils.knowledge_base import retrieve
from src.utils.history_manager import build_history_text, chunk_history_dynamic
from src.utils.ollama_client import OllamaClient

def build_prompt(query: str) -> str:
    # 1) Contexto manual
    user_ctx = st.session_state.get("context") or ""

    # 2) Contexto recuperado da KB (RAG-lite)
    kb = st.session_state.get("kb")
    recovered_text = ""
    meta = []
    if kb:
        recovered_text, meta = retrieve(query, kb, top_k=4, max_chars=4000)

    # 3) Histórico (texto plain concatenado, com chunking seguro)
    hist = st.session_state.get("history", [])
    _, total_chars, n_chunks, chunks = chunk_history_dynamic(hist, max_chars_per_chunk=8000, overlap=800)
    history_text = "\n\n".join(chunks) if chunks else ""

    # 4) Esforço/estilo da resposta
    effort = (st.session_state.get("effort") or "conciso").strip().lower()
    if effort == "exploratória":
        style = "resposta exploratória e criativa, mas objetiva"
    elif effort == "detalhada":
        style = "resposta detalhada e técnica, mantendo clareza"
    else:
        style = "resposta curta, direta e técnica"

    # 5) Prompt final (PT-BR, sem floreios)
    prompt = f"""Você é um assistente técnico que responde em português do Brasil, direto ao ponto, sem floreios e com humor sagaz quando couber.

# Contexto do usuário
{user_ctx.strip()}

# Trechos relevantes dos anexos (RAG)
{recovered_text.strip() or "[nenhum trecho relevante]"}

# Histórico (resumo bruto)
{history_text.strip() or "[início de conversa]"}

# Pergunta atual
{query.strip()}

# Exigências de estilo
- Seja {style}
- Se usar o material dos anexos, cite trechos/referências de forma natural dentro do texto
- Se faltar dado, diga o que falta em vez de inventar
"""
    return prompt

def answer_with_ollama(prompt: str) -> str:
    client = OllamaClient()
    model = st.session_state.get("model_choice") or "gpt-oss:20b"
    temperature = float(st.session_state.get("temperature") or 1.0)
    return client.ask(prompt=prompt, model=model, temperature=temperature)

# ======= Form de envio (compatível: limpa input, sem eco) =======
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Digite sua mensagem e pressione Enter:")
    submitted = st.form_submit_button("Enviar")

if submitted and user_input:
    # inclui pergunta no histórico
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state["messages"].append({"role": "user", "content": user_input, "ts": now})
    st.session_state["history"].append({"role": "user", "content": user_input})

    # monta prompt unificado (contexto + KB + histórico)
    try:
        prompt = build_prompt(user_input)
        reply = answer_with_ollama(prompt)
    except Exception as e:
        reply = f"Falha ao consultar o modelo local (Ollama). Detalhes: {e}"

    # inclui resposta
    st.session_state["messages"].append({"role": "assistant", "content": reply, "ts": datetime.now().strftime("%H:%M:%S")})
    st.session_state["history"].append({"role": "assistant", "content": reply})

    # rerun sem reusar o input (o form já limpou)
    st.experimental_rerun()

# Linha divisória (compat)
st.markdown("<hr>", unsafe_allow_html=True)

# GPU
show_gpu_info()
st.caption("🔹 GPT-OSS WebChat – Ambiente acelerado por GPU NVIDIA RTX 3050 Ti")
