python3 - <<'PY'
import numpy as np
a = np.array([[1, 2],[3, 4]], dtype=np.int8)
b = np.array([[10,20],[30,40]], dtype=np.int8)
a.tofile("input0.bin")
b.tofile("input1.bin")
print("sizes:", len(a.tobytes()), len(b.tobytes()))
PY

adb push input0.bin /data/local/tmp/et_vk/
adb push input1.bin /data/local/tmp/et_vk/

adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --inputs input0.bin,input1.bin'



# regenerate (float32), 4 elements => 16 bytes each
python3 - <<'PY'
import numpy as np
a = np.array([[1, 2],[3, 4]], dtype=np.float32)
b = np.array([[10,20],[30,40]], dtype=np.float32)
a.tofile("input0.bin")
b.tofile("input1.bin")
print("sizes:", len(a.tobytes()), len(b.tobytes()))
PY

adb push input0.bin /data/local/tmp/et_vk/
adb push input1.bin /data/local/tmp/et_vk/

# pass dtype if your runner supports it (pick one that --help shows)
adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --inputs input0.bin,input1.bin --input_dtype fp32'
# or
adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --inputs input0.bin,input1.bin --dtype float32'


python3 - <<'PY'
import os
print("input0.bin bytes:", os.path.getsize("input0.bin"))
print("input1.bin bytes:", os.path.getsize("input1.bin"))
PY
