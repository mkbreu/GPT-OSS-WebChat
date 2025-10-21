import tensorflow as tf
from tensorflow.python.client import device_lib

print("✅ GPUs detectadas:", tf.config.list_physical_devices('GPU'))

print("\n📦 Dispositivos disponíveis:")
for d in device_lib.list_local_devices():
    print(f"- {d.name} ({d.device_type}) | memória: {getattr(d, 'memory_limit', 'N/A')}")
