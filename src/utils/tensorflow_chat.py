"""Utilities for handling TensorFlow-backed text generation."""
from __future__ import annotations

# src/utils/tensorflow_chat.py
import importlib.util

import streamlit as st


def tensorflow_stack_available() -> bool:
    """Return True when tensorflow and transformers are importable."""
    tf_spec = importlib.util.find_spec("tensorflow")
    transformers_spec = importlib.util.find_spec("transformers")
    return tf_spec is not None and transformers_spec is not None


@st.cache_resource(show_spinner=False)
def _load_tf_stack(model_name: str):
    """Load tokenizer/model pair for the provided model name."""
    from transformers import AutoTokenizer, TFAutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def gpu_summary() -> str:
    """Return human readable description of GPU devices discovered by TensorFlow."""
    if not tensorflow_stack_available():
        return "TensorFlow/Transformers não instalados."
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "Nenhuma GPU detectada pelo TensorFlow (usando CPU)."
    names = ", ".join(device.name for device in gpus)
    return f"GPUs disponíveis: {names}"


def generate_with_tensorflow(
    prompt: str,
    *,
    model_name: str,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate a completion using a TensorFlow causal language model."""
    if not tensorflow_stack_available():
        return "*TensorFlow/Transformers não instalados neste ambiente.*"

    tokenizer, model = _load_tf_stack(model_name)
    inputs = tokenizer(prompt, return_tensors="tf")

    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(generation[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()
