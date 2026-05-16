cmake_minimum_required(VERSION 3.18)
project(HyperTensor C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HT_BUILD_CUDA    "Build CUDA kernels (requires CUDA toolkit)"  OFF)
option(HT_BUILD_VULKAN  "Build Vulkan compute backend"                OFF)
option(HT_BUILD_TESTS   "Build C test suite"                          OFF)
option(HT_BUILD_RUNTIME "Build the inference runtime"                 OFF)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Try to locate LAPACKE header/library (ships with OpenBLAS / lapacke-dev)
find_path(LAPACKE_INCLUDE_DIR lapacke.h
    HINTS /usr/include /usr/local/include /usr/include/openblas
          /opt/homebrew/include /opt/homebrew/opt/openblas/include)
find_library(LAPACKE_LIBRARY
    NAMES lapacke openblas
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
          /opt/homebrew/lib /opt/homebrew/opt/openblas/lib)

if(NOT LAPACKE_INCLUDE_DIR OR NOT LAPACKE_LIBRARY)
    message(FATAL_ERROR
        "LAPACKE not found.\n"
        "  Ubuntu/Debian:  sudo apt-get install liblapacke-dev\n"
        "  Fedora/RHEL:    sudo dnf install lapack-devel\n"
        "  macOS:          brew install openblas\n")
endif()

# ---------------------------------------------------------------------------
# libhypercore  (core geometric algorithms — GRC, GTC, FFN compression)
# ---------------------------------------------------------------------------
add_library(hypercore SHARED
    hypercore/hypercore.c
    lib/jury_gtc_kernel.c
)

target_include_directories(hypercore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hypercore>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib>
        ${LAPACKE_INCLUDE_DIR}
)

target_compile_options(hypercore PRIVATE
    -O3
    -ffast-math
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_compile_options(hypercore PRIVATE -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(hypercore PRIVATE -march=armv8-a+simd)
endif()

target_link_libraries(hypercore
    PUBLIC ${LAPACKE_LIBRARY} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} m
)

# On non-Windows, the shared lib name should be libhypercore.so / .dylib.
# Python ctypes will discover it via the name set here.
set_target_properties(hypercore PROPERTIES
    OUTPUT_NAME "hypercore"
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Runtime library  (inference engine — CPU tensor ops + JIT backends)
# ---------------------------------------------------------------------------
if(HT_BUILD_RUNTIME)
    file(GLOB RUNTIME_NN_SRCS      runtime/nn/*.c)
    file(GLOB RUNTIME_TENSOR_SRCS  runtime/tensor/*.c)
    file(GLOB RUNTIME_JIT_SRCS     runtime/jit/*.c)

    # Exclude CUDA and Vulkan sources unless the matching option is ON
    list(FILTER RUNTIME_NN_SRCS EXCLUDE REGEX ".*backend_cuda\\.c$")
    list(FILTER RUNTIME_NN_SRCS EXCLUDE REGEX ".*cuda_kernels\\.c$")
    if(NOT HT_BUILD_VULKAN)
        list(FILTER RUNTIME_NN_SRCS EXCLUDE REGEX ".*vulkan.*\\.c$")
    endif()

    add_library(ht_runtime STATIC
        ${RUNTIME_NN_SRCS}
        ${RUNTIME_TENSOR_SRCS}
        ${RUNTIME_JIT_SRCS}
    )

    target_include_directories(ht_runtime
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            runtime/nn
            runtime/tensor
            runtime/jit
            runtime/compute
            runtime/pseudocode
            ${LAPACKE_INCLUDE_DIR}
    )

    target_compile_options(ht_runtime PRIVATE
        -O3
        -ffast-math
        $<$<C_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-unused-parameter>
    )

    target_link_libraries(ht_runtime
        PUBLIC hypercore ${LAPACKE_LIBRARY} ${BLAS_LIBRARIES} m
    )

    # -----------------------------------------------------------------------
    # CUDA kernels (optional)
    # -----------------------------------------------------------------------
    if(HT_BUILD_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(ht_runtime PRIVATE
            runtime/nn/cuda_kernels.cu
            runtime/nn/backend_cuda.c
        )
        target_compile_definitions(ht_runtime PUBLIC HT_CUDA_ENABLED)
        target_link_libraries(ht_runtime PUBLIC CUDA::cudart CUDA::cublas)
    endif()

    # -----------------------------------------------------------------------
    # Vulkan compute (optional)
    # -----------------------------------------------------------------------
    if(HT_BUILD_VULKAN)
        find_package(Vulkan REQUIRED)
        target_sources(ht_runtime PRIVATE runtime/compute/vulkan_compute.c)
        target_compile_definitions(ht_runtime PUBLIC HT_VULKAN_ENABLED)
        target_link_libraries(ht_runtime PUBLIC Vulkan::Vulkan)
    endif()
endif()

# ---------------------------------------------------------------------------
# C test suite
# ---------------------------------------------------------------------------
if(HT_BUILD_TESTS)
    enable_testing()

    foreach(test_src
        tests/runtime/test_kernels.c
        tests/runtime/test_model_meta.c
        tests/runtime/test_chat.c
        tests/runtime/test_tokenizer.c
    )
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE
            $<$<BOOL:${HT_BUILD_RUNTIME}>:ht_runtime>
            hypercore
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # eval_harness lives directly under tests/
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/eval_harness.c")
        add_executable(eval_harness tests/eval_harness.c)
        target_link_libraries(eval_harness PRIVATE hypercore)
        add_test(NAME eval_harness COMMAND eval_harness)
    endif()
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
install(TARGETS hypercore
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(FILES
    hypercore/hypercore.h
    lib/jury_gtc_kernel.c
    DESTINATION include/hypertensor
)
if(HT_BUILD_RUNTIME)
    install(TARGETS ht_runtime
        ARCHIVE DESTINATION lib
    )
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "HyperTensor build configuration:")
message(STATUS "  LAPACKE:  ${LAPACKE_LIBRARY}")
message(STATUS "  BLAS:     ${BLAS_LIBRARIES}")
message(STATUS "  CUDA:     ${HT_BUILD_CUDA}")
message(STATUS "  Vulkan:   ${HT_BUILD_VULKAN}")
message(STATUS "  Runtime:  ${HT_BUILD_RUNTIME}")
message(STATUS "  Tests:    ${HT_BUILD_TESTS}")
message(STATUS "")
