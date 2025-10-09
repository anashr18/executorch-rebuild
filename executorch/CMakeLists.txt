cmake_minimum_required(VERSION 3.16)
project(executorch_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(executorch_runtime STATIC
    runtime/core/json.cpp
    runtime/executor/program.cpp
    runtime/executor/executor.cpp
)

target_include_directories(executorch_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(executor_runner
    examples/executor_runner/main.cpp
)

target_link_libraries(executor_runner PRIVATE executorch_runtime)
