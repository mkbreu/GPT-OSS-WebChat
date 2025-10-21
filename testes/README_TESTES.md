📄 README_TESTES.md
# 🧩 Testes de GPU — TensorFlow + CUDA

Esta pasta contém scripts de diagnóstico e benchmark para confirmar se o **TensorFlow** está utilizando corretamente a **GPU NVIDIA** (aceleração de hardware).

---

## 📁 Estrutura



testes/
├── test_gpu.py
├── benchmark_gpu.py
├── benchmark_visual.py
└── README_TESTES.md


---

## 🧠 Objetivo

Verificar:
1. Se o TensorFlow reconhece sua GPU corretamente.  
2. Se a aceleração por GPU está realmente ativa.  
3. Qual é o ganho de desempenho (CPU x GPU) em operações matemáticas intensas.  
4. (Opcional) Gerar um **gráfico visual comparativo** de desempenho.

---

## ▶️ Execução dos Testes

Antes de executar, **ative o ambiente virtual**:
```powershell
.\venv-tf\Scripts\Activate.ps1
cd testes

1️⃣ Teste de detecção da GPU
python test_gpu.py


Saída esperada:

✅ GPUs detectadas: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

📦 Dispositivos disponíveis:
- /device:CPU:0 (CPU)
- /device:GPU:0 (GPU)


Se a GPU aparecer na lista → o TensorFlow está configurado corretamente ✅

2️⃣ Benchmark simples (CPU x GPU)
python benchmark_gpu.py


Saída esperada (exemplo):

🚀 Teste de desempenho TensorFlow (matriz 10000x10000)
🧠 CPU: 2.945 segundos
💻 GPU: 1.971 segundos
⚡ Aceleração: 1.5x mais rápido com GPU


💡 O ganho de desempenho pode variar conforme o modelo da GPU, carga do sistema e tamanho da matriz.

3️⃣ Benchmark visual (opcional)
python benchmark_visual.py


Exibe um gráfico comparativo CPU x GPU com diferentes tamanhos de matrizes (de 1.000 a 20.000).
O gráfico mostra o ponto a partir do qual a GPU começa a superar a CPU significativamente.

📦 Requisitos de dependência

Certifique-se de que seu ambiente virtual tenha os seguintes pacotes:

pip install tensorflow==2.10.1 matplotlib==3.9.2

⚙️ Interpretação dos resultados
Resultado	Significado
GPU listada e benchmark mais rápido	Aceleração GPU funcionando corretamente ✅
GPU não listada / erro “Could not load cudnn64_8.dll”	Verifique instalação do cuDNN e variável PATH
Tempos semelhantes CPU e GPU	GPU pode estar em modo econômico ou operação não paralelizável
🔍 Arquivos auxiliares
Arquivo	Função
test_gpu.py	Detecta dispositivos TensorFlow (CPU e GPU)
benchmark_gpu.py	Executa teste comparativo direto CPU x GPU
benchmark_visual.py	(Opcional) Gera gráfico CPU x GPU com Matplotlib
📘 Observação importante

Este diretório serve apenas para testes e diagnóstico.
Não é necessário em produção, mas é essencial para validar se a configuração CUDA + cuDNN + TensorFlow GPU está estável antes de usar o app principal (app.py).

Desenvolvido por Maykon Abreu — GPT-OSS-WebChat