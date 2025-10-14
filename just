which glslc && glslc --version   # must work (from LunarG SDK or NDK)

rm -rf out/android-arm64
cmake -S . -B out/android-arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DGLSLC_EXECUTABLE="$(which glslc)" \
  -DBINDING_GEN_EXE="$PWD/out/host/tools/bindgen/binding_gen_host" \
  -DCMAKE_BUILD_TYPE=Release

# first pass single-job to confirm shader rules are there
cmake --build out/android-arm64 -v -j1
# then go parallel
cmake --build out/android-arm64 -j"$(nproc)"





# In the repo:
find executorch -ipath "*backends*/*vulkan*/*glsl*" -type f -maxdepth 6

# In the build dir, some projects copy/pack shaders:
find out/android-arm64 -iname "*.glsl" -o -iname "*_spv.cpp"




git submodule sync
git submodule update --init --recursive
git status