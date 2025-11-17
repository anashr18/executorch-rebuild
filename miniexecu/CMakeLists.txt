cmake_minimum_required(VERSION 3.16)
project(mini_executorch_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mini_exec
    src/main.cpp
    src/kernels_cpu.cpp
)

target_include_directories(mini_exec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
