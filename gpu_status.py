import streamlit as st
import tensorflow as tf
from tensorflow.python.client import device_lib

st.set_page_config(page_title="Status da GPU", page_icon="⚡", layout="centered")

st.title("⚡ Diagnóstico de GPU e Aceleração de Hardware")

try:
    devices = device_lib.list_local_devices()
    gpu_devices = [d for d in devices if d.device_type == "GPU"]

    if gpu_devices:
        gpu = gpu_devices[0]
        st.success("✅ GPU detectada e ativa!")
        st.write(f"**Nome:** {gpu.physical_device_desc.split('name: ')[1].split(',')[0]}")
        st.write(f"**Tipo:** {gpu.device_type}")
        st.write(f"**Memória disponível:** {round(gpu.memory_limit / (1024**3), 2)} GB")

        cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'Desconhecida')
        cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'Desconhecida')
        st.write(f"**CUDA:** {cuda_version}")
        st.write(f"**cuDNN:** {cudnn_version}")

        st.info("A aceleração de hardware está configurada corretamente.")
    else:
        st.warning("⚠️ Nenhuma GPU detectada. TensorFlow está usando apenas CPU.")
        st.write("Verifique drivers NVIDIA e pacotes CUDA/cuDNN.")

    st.subheader("🧩 Dispositivos locais detectados:")
    for device in devices:
        st.code(str(device), language="bash")

except Exception as e:
    st.error(f"Erro ao verificar status da GPU: {str(e)}")
