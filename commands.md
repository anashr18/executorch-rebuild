cmake -S executorch -B executorch/cmake-out -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake -S . -B cmake-out -DCMAKE_EXPORT_COMPILE_COMMANDS=ON