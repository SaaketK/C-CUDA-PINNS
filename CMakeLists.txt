cmake_minimum_required(VERSION 3.20)
project(PINN C)
include(CTest)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OPENBLAS QUIET IMPORTED_TARGET openblas)
endif()
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
    list(APPEND PINN_CUDA_SOURCES
        src/core/ops_cuda.cu
        src/nn/optimizer_cuda.cu)
endif()

function(pinn_configure_target target)
    if(TARGET PkgConfig::OPENBLAS)
        target_compile_definitions(${target} PRIVATE PINN_USE_OPENBLAS)
        target_link_libraries(${target} PRIVATE PkgConfig::OPENBLAS)
    endif()
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
        src/nn/activations.c
        src/nn/optimizer_cpu.c
        src/nn/optimizer.c
        src/nn/lbfgs.c)

set(PINN_AUTODIFF_SOURCES
        src/autodiff/jet.c)

set(PINN_SAMPLER_SOURCES
        src/pinn/sampler.c)

set(PINN_RESIDUAL_SOURCES
        src/pinn/residual.c)

set(PINN_MODEL_IO_SOURCES
        src/surrogate/model_io.c)

set(PINN_HEAT1D_SURROGATE_SOURCES
        src/surrogate/heat1d_surrogate.c
        src/surrogate/heat1d_modal_surrogate.c)

add_library(pinn_heat1d_backend SHARED
        src/surrogate/heat1d_inference.c
        src/surrogate/heat1d_modal_surrogate.c
        ${PINN_MODEL_IO_SOURCES}
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES})
set_target_properties(pinn_heat1d_backend PROPERTIES
        OUTPUT_NAME pinn_heat1d_backend)
pinn_configure_target(pinn_heat1d_backend)
target_include_directories(pinn_heat1d_backend PUBLIC include)

option(PINN_BUILD_DEVELOPMENT_TARGETS
        "Build training programs, benchmarks, and tests" ON)

if(PINN_BUILD_DEVELOPMENT_TARGETS)

add_executable(core_test
        tests/core-test.c
        ${PINN_CORE_SOURCES})

add_executable(nn_test
        tests/nn-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(gradcheck_test
        tests/gradcheck-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(jet_test
        tests/jet-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(heat_residual_test
        tests/heat-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(heat1d_parametric_test
        tests/heat1d-parametric-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_RESIDUAL_SOURCES}
        ${PINN_HEAT1D_SURROGATE_SOURCES})

add_executable(model_io_test
        tests/model-io-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_MODEL_IO_SOURCES})

add_executable(heat1d_inference_test
        tests/heat1d-inference-test.c)
target_link_libraries(heat1d_inference_test PRIVATE pinn_heat1d_backend m)
target_compile_definitions(heat1d_inference_test PRIVATE
        PINN_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/heat1d/v1/model.pinn")

add_executable(black_scholes_residual_test
        tests/black-scholes-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(black_scholes1d_train
        projects/black-scholes1d/black_scholes1d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(heat1d_train
        projects/heat1d/heat1d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES}
        ${PINN_MODEL_IO_SOURCES}
        ${PINN_HEAT1D_SURROGATE_SOURCES})

add_executable(heat2d_train
        projects/heat2d/heat2d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(sampler_test
        tests/sampler-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_SAMPLER_SOURCES})

add_executable(ops_forward_test
        tests/ops-forward-test.c
        ${PINN_CORE_SOURCES})

add_executable(ops_benchmark
        tests/ops-benchmark.c
        ${PINN_CORE_SOURCES})

pinn_configure_target(core_test)
pinn_configure_target(nn_test)
pinn_configure_target(gradcheck_test)
pinn_configure_target(jet_test)
pinn_configure_target(heat_residual_test)
pinn_configure_target(heat1d_parametric_test)
pinn_configure_target(model_io_test)
pinn_configure_target(heat1d_inference_test)
pinn_configure_target(black_scholes_residual_test)
pinn_configure_target(black_scholes1d_train)
pinn_configure_target(heat1d_train)
pinn_configure_target(heat2d_train)
pinn_configure_target(sampler_test)
pinn_configure_target(ops_forward_test)
pinn_configure_target(ops_benchmark)

target_include_directories(core_test PRIVATE include)
target_include_directories(nn_test PRIVATE include)
target_include_directories(gradcheck_test PRIVATE include)
target_include_directories(jet_test PRIVATE include)
target_include_directories(heat_residual_test PRIVATE include)
target_include_directories(heat1d_parametric_test PRIVATE include)
target_include_directories(model_io_test PRIVATE include)
target_include_directories(heat1d_inference_test PRIVATE include)
target_include_directories(black_scholes_residual_test PRIVATE include)
target_include_directories(black_scholes1d_train PRIVATE include)
target_include_directories(heat1d_train PRIVATE include)
target_include_directories(heat2d_train PRIVATE include)
target_include_directories(sampler_test PRIVATE include)
target_include_directories(ops_forward_test PRIVATE include)
target_include_directories(ops_benchmark PRIVATE include)
add_test(NAME heat1d_parametric_test COMMAND heat1d_parametric_test)
add_test(NAME model_io_test COMMAND model_io_test)
add_test(NAME heat1d_inference_test COMMAND heat1d_inference_test)

endif()
