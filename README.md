# GPT-OSS-WebChat

Aplicação web em **Streamlit** para conversar com **modelos locais** via **Ollama**, com suporte à **GPU (TensorFlow + CUDA + cuDNN)**, leitura de anexos e RAG-lite. Mantém histórico (TXT/DOCX), controla tamanho de contexto e roda 100% offline.

> **Recomendação forte:** salve este projeto em **Documentos** (Windows) para padronizar os caminhos.
> Exemplo: `C:\Users\<SEU_USUARIO>\Documents\GPT-OSS-WebChat`

---

## Recursos

* **Modelos locais (Ollama)**, sem custo de API
* **RAG-lite** sobre anexos (CSV, Excel, TXT, DOCX, PDF, PPTX/ODP, imagens)
* **Histórico exportável** (TXT + DOCX)
* **Compatível com Streamlit legado** (sem `st.chat_message`/`st.divider`)
* **GPU** com TensorFlow 2.10.1 (Windows)

---

## Estrutura

```
GPT-OSS-WebChat/
├─ app.py
├─ requirements.txt
├─ README.md
├─ conversations/
│  └─ exports/
├─ src/
│  ├─ components/
│  │  └─ sidebar.py
│  └─ utils/
│     ├─ file_reader.py
│     ├─ knowledge_base.py
│     ├─ history_manager.py
│     └─ ollama_client.py
└─ testes/
   ├─ test_gpu.py
   ├─ benchmark_gpu.py
   └─ benchmark_visual.py
```

---

## Pré-requisitos (Windows)

* **Python 3.10/3.11** (x64) instalado no PATH
* **CUDA 11.2** + **cuDNN 8.1.1** para TensorFlow 2.10.1
* **Ollama** instalado e em execução
* **Git** e **VS Code** (opcional, mas recomendado)

**Observações rápidas**

* TensorFlow GPU no Windows é **2.10.1** (versões >2.10 não dão GPU no Windows)
* Combine **CUDA 11.2** com **cuDNN 8.1.1**
* Se não tiver CUDA/cuDNN, primeiro rode com CPU; depois habilite GPU

---

## Instalação (VS Code ou PowerShell)

> Supondo que você salvou o projeto em **Documentos**:
> `C:\Users\<SEU_USUARIO>\Documents\GPT-OSS-WebChat`

1. **Abra o VS Code** e o **Terminal integrado** (ou abra o PowerShell)

2. **Vá até a pasta do projeto**

```powershell
cd "C:\Users\<SEU_USUARIO>\Documents\GPT-OSS-WebChat"
```

3. **Crie e ative o ambiente virtual**

```powershell
python -m venv venv-tf
.\venv-tf\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4. **Instale as dependências**

```powershell
pip install -r requirements.txt
```

5. **(Opcional) Instale CUDA/cuDNN para GPU**

* CUDA 11.2: [https://developer.nvidia.com/cuda-11.2.0-download-archive](https://developer.nvidia.com/cuda-11.2.0-download-archive)
  Instala em: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`
* cuDNN 8.1.1 (para CUDA 11.2): [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
  Extraia e copie para as pastas `bin`, `include`, `lib\x64` da instalação CUDA

6. **Ativar GPU no TensorFlow (se CUDA/cuDNN instalados)**

```powershell
pip install tensorflow==2.10.1
```

7. **Ativar embeddings no Ollama (melhora precisão e velocidade do RAG)**

```powershell
ollama pull nomic-embed-text
```

> Se quiser GPU no Ollama: iniciar o servidor com variável de ambiente

```powershell
$env:OLLAMA_USE_GPU=1
ollama serve
```

Depois, mantenha o Ollama rodando e use o app.

---

## Rodando a aplicação

Com o ambiente ativado dentro da pasta do projeto:

```powershell
streamlit run app.py
```

A interface abre no navegador.
Fluxo de uso:

1. Configure **modelo**, **temperatura** e **tamanho de contexto** na sidebar
2. **Anexe arquivos** (serão indexados e divididos em trechos)
3. Digite sua pergunta; o app junta **Contexto + Anexos + Histórico** numa única prompt
4. Exporte o histórico quando quiser (TXT/DOCX)

---

## Mensagem “Embeddings indisponíveis (fallback: palavras-chave)”

Se, ao anexar arquivos, a sidebar mostrar:

```
Base preparada (... trechos). Embeddings indisponíveis (fallback: palavras-chave).
```

significa que o app caiu no modo simples (busca por palavras-chave) porque não conseguiu gerar **embeddings**.
Correção:

```powershell
# 1) Verifique se o Ollama está rodando
ollama list

# 2) Se não houver 'nomic-embed-text', instale:
ollama pull nomic-embed-text

# 3) (Opcional) Usar GPU no Ollama
$env:OLLAMA_USE_GPU=1
ollama serve

# 4) Recarregue o app (streamlit)
```

Quando os embeddings estiverem ativos, você verá:

```
Base preparada (... trechos). Embeddings: ok.
```

---

## Verificação rápida da GPU (TensorFlow)

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

Se listar algo como `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`, a GPU está ativa.

---

## Performance

* **Token context**: 8k funciona bem. Se ficar lento, reduza um pouco.
* **RAG-lite**: por padrão recupera até ~4k caracteres dos anexos; ajuste se necessário.
* **Modelo**: modelos menores no Ollama (ex.: `llama3:8b`, `phi3:3.8b`) respondem mais rápido.

---

## Licença

MIT

---

## Créditos

Arquitetura e implementação por Maykon Abreu.
Baseado no design modular do GPT-OSS WebChat.
