# adjust these paths to your build/model
RUNNER=out/android-arm64/bin/vulkan_executor_runner
MODEL=/path/to/your/model.pte

# (optional) shrink the runner a bit if you want
# $ANDROID_NDK must be set
"$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$RUNNER" 2>/dev/null || true


adb shell 'rm -rf /data/local/tmp/et_vk && mkdir -p /data/local/tmp/et_vk'
adb push "$RUNNER" /data/local/tmp/et_vk/
adb push "$MODEL"  /data/local/tmp/et_vk/model.pte

# if you have *.so later (dynamic build), uncomment:
# adb push out/android-arm64/lib /data/local/tmp/et_vk/lib
# adb push "$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" /data/local/tmp/et_vk/lib/




adb shell 'cd /data/local/tmp/et_vk && chmod +x vulkan_executor_runner && ./vulkan_executor_runner --model model.pte'



adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --help' || true


python3 - <<'PY'
import numpy as np
# adjust shape/dtype to your model
a = np.ones((1,3,224,224), dtype=np.float32)
a.tofile("input.bin")
PY

adb push input.bin /data/local/tmp/et_vk/
adb shell 'cd /data/local/tmp/et_vk && ./vulkan_executor_runner --model model.pte --input input.bin'

