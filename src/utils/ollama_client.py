# src/utils/ollama_client.py
import requests
from typing import Dict, List

class OllamaClient:
    """
    Wrapper simples para a API local do Ollama.
    Compatível com /api/generate e fallback para /api/chat.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def ask(self, *, prompt: str, model: str, temperature: float = 1.0) -> str:
        # 1) Tenta /api/generate (não-stream)
        gen_payload: Dict = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
        }
        try:
            r = requests.post(f"{self.host}/api/generate", json=gen_payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # respostas típicas: {"response": "..."} ou {"output": "..."}
            return data.get("response") or data.get("output") or str(data)
        except Exception:
            # 2) Fallback: /api/chat com messages
            chat_payload: Dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
                "stream": False,
            }
            r = requests.post(f"{self.host}/api/chat", json=chat_payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # formatos variam
            return data.get("message", {}).get("content") or data.get("response") or data.get("text") or str(data)

    def list_models(self) -> List[str]:
        endpoints = ["/api/tags", "/api/models"]
        for ep in endpoints:
            try:
                r = requests.get(f"{self.host}{ep}", timeout=8)
                r.raise_for_status()
                j = r.json()
                if isinstance(j, dict):
                    if "models" in j and isinstance(j["models"], list):
                        names = []
                        for m in j["models"]:
                            if isinstance(m, dict) and "name" in m:
                                names.append(m["name"])
                            elif isinstance(m, str):
                                names.append(m)
                        if names:
                            return names
                    if "tags" in j and isinstance(j["tags"], list):
                        return [t.get("name") if isinstance(t, dict) else str(t) for t in j["tags"]]
                    if isinstance(j, list):
                        return [i.get("name") if isinstance(i, dict) and "name" in i else str(i) for i in j]
                elif isinstance(j, list):
                    return [i.get("name") if isinstance(i, dict) and "name" in i else str(i) for i in j]
            except Exception:
                continue
        raise RuntimeError("Nenhum modelo listado pelo Ollama (verifique se o serviço está rodando).")
