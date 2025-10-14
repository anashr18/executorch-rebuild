python3 - <<'PY'
import numpy as np

# 2x2 test tensors
a = np.array([[1, 2],
              [3, 4]], dtype=np.float32)
b = np.array([[10, 20],
              [30, 40]], dtype=np.float32)

a.tofile("input0.bin")
b.tofile("input1.bin")

print("Input A:\n", a)
print("Input B:\n", b)
PY



adb push input0.bin /data/local/tmp/et_vk/
adb push input1.bin /data/local/tmp/et_vk/



adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --inputs input0.bin,input1.bin'


adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --input0 input0.bin --input1 input1.bin'


adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --input input0.bin --input input1.bin'

