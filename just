export ANDROID_NDK=/abs/path/android-ndk-r27b
export ANDROID_ABI=arm64-v8a

HOST_GEN="$PWD/out/host/tools/bindgen/binding_gen_host"         # adjust if your tool name/path differs

rm -rf out/android-arm64
cmake -S . -B out/android-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM=android-24 \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DBINDING_GEN_EXE="$HOST_GEN" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build out/android-arm64 -j"$(nproc)" --target install





# confirm the flag actually took:
grep -E 'EXECUTORCH_BUILD_VULKAN.*=ON' out/android-arm64/CMakeCache.txt

# list targets; you should see vulkan ones (e.g., vulkan_executor_runner)
cmake --build out/android-arm64 --target help | grep -i vulkan

# find built vulkan bits
find out/android-arm64 -iname '*vulkan*'
