import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# Lista de tamanhos de matrizes a serem testadas
sizes = [1000, 3000, 5000, 7000, 10000, 15000, 20000]
cpu_times, gpu_times = [], []

print("\n📊 Benchmark visual CPU x GPU - TensorFlow\n")

for n in sizes:
    print(f"🔹 Testando matrizes {n}x{n}...")

    # CPU
    with tf.device('/CPU:0'):
        a = tf.random.normal([n, n])
        b = tf.random.normal([n, n])
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        cpu_times.append(time.time() - start)

    # GPU
    with tf.device('/GPU:0'):
        a = tf.random.normal([n, n])
        b = tf.random.normal([n, n])
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        gpu_times.append(time.time() - start)

# Exibir resultados numéricos
print("\n📈 Resultados detalhados:\n")
for i, n in enumerate(sizes):
    fator = cpu_times[i] / gpu_times[i] if gpu_times[i] > 0 else float('inf')
    print(f"Matriz {n}x{n}: 🧠 CPU: {cpu_times[i]:.3f}s | 💻 GPU: {gpu_times[i]:.3f}s | ⚡ {fator:.2f}x mais rápido")

# Gerar gráfico comparativo
plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_times, marker='o', label='CPU')
plt.plot(sizes, gpu_times, marker='o', label='GPU')
plt.title("Comparativo de Desempenho TensorFlow: CPU x GPU")
plt.xlabel("Tamanho da matriz (n x n)")
plt.ylabel("Tempo de execução (segundos)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
