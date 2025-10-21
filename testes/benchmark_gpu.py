import tensorflow as tf
import time

# Força TensorFlow a não usar GPU para o primeiro teste
def run_on_cpu():
    with tf.device('/CPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

# Força uso da GPU
def run_on_gpu():
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

print("\n🚀 Teste de desempenho TensorFlow (matriz 10000x10000)")

cpu_time = run_on_cpu()
gpu_time = run_on_gpu()

print(f"\n🧠 CPU: {cpu_time:.3f} segundos")
print(f"💻 GPU: {gpu_time:.3f} segundos")
print(f"\n⚡ Aceleração: {cpu_time / gpu_time:.1f}x mais rápido com GPU")
