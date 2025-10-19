# src/utils/ollama_client.py
import requests
from typing import Dict, List, Optional

class OllamaClient:
    """
    Wrapper simples para a API local do Ollama.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def ask(self, *, prompt: str, model: str, temperature: float = 1.0) -> str:
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Adaptamos ao formato esperado; alguns Ollama retornam "response" ou "text"
        return data.get("response") or data.get("text") or str(data)

    def list_models(self) -> List[str]:
        """
        Retorna lista de modelos instalados. Tenta '/api/tags' e '/api/models'.
        """
        endpoints = ["/api/tags", "/api/models"]
        for ep in endpoints:
            try:
                r = requests.get(f"{self.host}{ep}", timeout=8)
                r.raise_for_status()
                j = r.json()
                # Formatos possíveis:
                # {"models":[{"name":"gpt-20b"}, ...]}
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
                    # Alguns endpoints retornam tags or simple lists
                    if "tags" in j and isinstance(j["tags"], list):
                        return [t.get("name") if isinstance(t, dict) else str(t) for t in j["tags"]]
                    # Se for já uma lista simples
                    if isinstance(j, list):
                        return [i.get("name") if isinstance(i, dict) and "name" in i else str(i) for i in j]
            except Exception:
                # tenta próximo endpoint
                continue
        # Se não encontrou nada, lança para que o caller trate
        raise RuntimeError("Nenhum modelo listado pelo Ollama (verifique se o serviço está rodando).")
