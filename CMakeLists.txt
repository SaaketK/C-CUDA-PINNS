cmake_minimum_required(VERSION 3.20)
project(PINN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(CUDAToolkit QUIET)

include(CheckLanguage)

if(CUDAToolkit_FOUND AND NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()

check_language(CUDA)

set(PINN_CUDA_SOURCES)
if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND PINN_CUDA_SOURCES src/core/ops_cuda.cu)
endif()

function(pinn_configure_target target)
    if(OpenMP_C_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    endif()

    if(CUDAToolkit_FOUND)
        target_compile_definitions(${target} PRIVATE PINN_USE_CUDA)
        target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas)
    endif()
endfunction()

set(PINN_CORE_SOURCES
        src/core/tensor.c
        src/core/ops_cpu.c
        src/core/ops.c
        src/core/autograd.c
        src/core/backend.c
        ${PINN_CUDA_SOURCES})

set(PINN_NN_SOURCES
        src/nn/mlp.c
        src/nn/optimizer.c
        src/nn/lbfgs.c)

set(PINN_AUTODIFF_SOURCES
        src/autodiff/jet.c)

set(PINN_SAMPLER_SOURCES
        src/pinn/sampler.c)

set(PINN_RESIDUAL_SOURCES
        src/pinn/residual.c)

add_executable(pinn_main
        src/main.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(core_test
        src/tests/core-test.c
        ${PINN_CORE_SOURCES})

add_executable(nn_test
        src/tests/nn-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(gradcheck_test
        src/tests/gradcheck-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(jet_test
        src/tests/jet-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(heat_residual_test
        src/tests/heat-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(black_scholes_residual_test
        src/tests/black-scholes-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(black_scholes1d_train
        examples/blackscholes1d/black_scholes1d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(heat1d_train
        examples/heat1d/heat1d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(heat2d_train
        examples/heat2d/heat2d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(sampler_test
        src/tests/sampler-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_SAMPLER_SOURCES})

add_executable(ops_forward_test
        src/tests/ops-forward-test.c
        ${PINN_CORE_SOURCES})

add_executable(ops_benchmark
        src/tests/ops-benchmark.c
        ${PINN_CORE_SOURCES})

pinn_configure_target(pinn_main)
pinn_configure_target(core_test)
pinn_configure_target(nn_test)
pinn_configure_target(gradcheck_test)
pinn_configure_target(jet_test)
pinn_configure_target(heat_residual_test)
pinn_configure_target(black_scholes_residual_test)
pinn_configure_target(black_scholes1d_train)
pinn_configure_target(heat1d_train)
pinn_configure_target(sampler_test)
pinn_configure_target(ops_forward_test)
pinn_configure_target(ops_benchmark)

target_include_directories(pinn_main PRIVATE include)
target_include_directories(core_test PRIVATE include)
target_include_directories(nn_test PRIVATE include)
target_include_directories(gradcheck_test PRIVATE include)
target_include_directories(jet_test PRIVATE include)
target_include_directories(heat_residual_test PRIVATE include)
target_include_directories(black_scholes_residual_test PRIVATE include)
target_include_directories(black_scholes1d_train PRIVATE include)
target_include_directories(heat1d_train PRIVATE include)
target_include_directories(heat2d_train PRIVATE include)
target_include_directories(sampler_test PRIVATE include)
target_include_directories(ops_forward_test PRIVATE include)
target_include_directories(ops_benchmark PRIVATE include)
