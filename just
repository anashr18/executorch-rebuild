# replace this with the FIRST path you saw in your find output
SHADER_DIR="/abs/path/to/executorch/backends/vulkan/runtime/graph/ops/glsl"

# does the key shader exist here (or nearby)?
test -f "$SHADER_DIR/graph_ops.glsl" || find "$SHADER_DIR" -maxdepth 2 -name "graph_ops.glsl" -print

# list a few shaders to be sure
find "$SHADER_DIR" -maxdepth 1 -name "*.glsl" | head



rm -rf out/android-arm64
cmake -S . -B out/android-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DGLSLC_EXECUTABLE="$(which glslc)" \
  -DVULKAN_COMPUTE_SHADERS_DIR="$SHADER_DIR" \
  -DEXECUTORCH_VULKAN_SHADER_DIR="$SHADER_DIR" \
  -DET_VK_SHADER_DIR="$SHADER_DIR" \
  -DCMAKE_BUILD_TYPE=Release