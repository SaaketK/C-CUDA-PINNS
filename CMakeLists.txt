cmake_minimum_required(VERSION 3.20)
project(PINN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(pinn_autograd_demo
        src/main.c
        src/core/tensor.c
        src/core/ops.c
        src/core/autograd.c)

target_include_directories(pinn_autograd_demo PRIVATE include)
