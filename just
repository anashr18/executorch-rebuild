grep -RniE "ShaderLibrary|ComputeShaders_spv|graph[_-]ops\.glsl|glslc" cmake/ executorch backends | head -n 50
find executorch -type d -iname glsl | sed -n '1,50p'


-DVULKAN_COMPUTE_SHADERS_DIR="$PWD/executorch/backends/vulkan/runtime/graph/ops/glsl"
# or
-DEXECUTORCH_VULKAN_SHADER_DIR="$PWD/executorch/backends/vulkan/runtime/graph/ops/glsl"