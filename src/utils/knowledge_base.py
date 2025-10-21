# src/utils/knowledge_base.py

from __future__ import annotations
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
from pathlib import Path

from src.utils.file_reader import (
    read_csv_file, read_excel_file, read_txt_file, read_docx_file,
    read_pdf_file, read_pptx_file, read_image_file,
)

# Parâmetros padrão (simples)
CHUNK_SIZE_CHARS = 3000
CHUNK_OVERLAP_CHARS = 400
EMBED_MODEL = "nomic-embed-text"  # `ollama pull nomic-embed-text`
TOP_K = 4
MAX_RETRIEVED_CHARS = 4000

@dataclass
class KBChunk:
    text: str
    meta: Dict[str, Any]

@dataclass
class KnowledgeBase:
    chunks: List[KBChunk]
    vectors: Optional[np.ndarray]
    use_embeddings: bool
    meta: Dict[str, Any]  # ex.: {"embed_model": "...", "file_sigs": [...]}

_WS = re.compile(r"\s+")

def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(_WS, " ", s).strip()
    return s

def _chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def _embed_ollama(texts: List[str], host: str = "http://localhost:11434", model: str = EMBED_MODEL, timeout: int = 60) -> np.ndarray:
    vectors = []
    url = f"{host.rstrip('/')}/api/embeddings"
    for t in texts:
        payload = {"model": model, "prompt": t}
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        vec = j.get("embedding")
        if not vec:
            raise RuntimeError("Resposta de embeddings sem vetor.")
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def _keyword_score(query: str, texts: List[str]) -> np.ndarray:
    def toks(x: str) -> set:
        import re as _re
        return set(_re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", x.lower()))
    q = toks(query)
    return np.array([len(q & toks(t)) for t in texts], dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm

def _read_any_file(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".csv":
        return read_csv_file(data)
    if ext in {".xlsx", ".xls", ".ods"}:
        return read_excel_file(data)
    if ext in {".txt", ".md", ".rtf"}:
        return read_txt_file(data)
    if ext == ".docx":
        return read_docx_file(data)
    if ext == ".pdf":
        return read_pdf_file(data)
    if ext in {".pptx", ".odp"}:
        return read_pptx_file(data)
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif"}:
        return f"[Imagem anexada: {name}]"
    return f"[Arquivo {name} ({ext}) não suportado]"

def _fingerprint(name: str, data: bytes) -> str:
    h = hashlib.sha1()
    h.update(name.encode("utf-8"))
    h.update(data)
    return h.hexdigest()

def build_kb_from_uploads(uploaded_files: List) -> KnowledgeBase:
    chunks: List[KBChunk] = []
    file_sigs: List[str] = []

    for f in uploaded_files or []:
        try:
            raw = f.getvalue()
            sig = _fingerprint(f.name, raw)
            file_sigs.append(sig)
            text = _read_any_file(f.name, raw)
            for i, c in enumerate(_chunk_text(text)):
                chunks.append(KBChunk(text=c, meta={"file": f.name, "chunk_id": i, "sig": sig}))
        except Exception as e:
            chunks.append(KBChunk(text=f"[ERRO ao ler {f.name}: {e}]", meta={"file": f.name, "chunk_id": 0}))

    if not chunks:
        return KnowledgeBase(chunks=[], vectors=None, use_embeddings=False, meta={"embed_model": None, "file_sigs": []})

    texts = [c.text for c in chunks]

    try:
        vecs = _embed_ollama(texts)
        use_emb = True
    except Exception:
        vecs = None
        use_emb = False

    return KnowledgeBase(
        chunks=chunks,
        vectors=vecs,
        use_embeddings=use_emb,
        meta={"embed_model": EMBED_MODEL if use_emb else None, "file_sigs": file_sigs, "processed": True}
    )

def retrieve(query: str, kb: KnowledgeBase, top_k: int = TOP_K, max_chars: int = MAX_RETRIEVED_CHARS) -> Tuple[str, List[Dict[str, Any]]]:
    if not kb or not kb.chunks:
        return "", []

    texts = [c.text for c in kb.chunks]

    if kb.use_embeddings and kb.vectors is not None:
        qv = _embed_ollama([query])[0]
        sims = _cosine_sim(qv, kb.vectors)
    else:
        sims = _keyword_score(query, texts)

    idx = np.argsort(-sims)[:max(top_k, 1)]
    picked = [kb.chunks[i] for i in idx]

    out_parts, used, meta_list = [], 0, []
    for ch in picked:
        part = ch.text.strip()
        remain = max_chars - used
        if remain <= 0:
            break
        if len(part) > remain:
            part = part[:remain] + " …"
        out_parts.append(f"[{ch.meta.get('file')}, chunk {ch.meta.get('chunk_id')}] {part}")
        used += len(part)
        meta_list.append(ch.meta)

    return "\n\n".join(out_parts), meta_list
