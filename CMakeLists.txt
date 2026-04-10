cmake_minimum_required(VERSION 3.16)
project(HyperTensor VERSION 0.4.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Sources ──────────────────────────────────────────────────────────────────

set(HYPERTENSOR_SOURCES
    host/hal.c
    host/main.c
    runtime/nn/llm.c
    runtime/nn/gguf.c
    runtime/jit/x86_jit.c
    runtime/jit/llm_jit.c
)

add_executable(hypertensor ${HYPERTENSOR_SOURCES})

# ── Include paths ────────────────────────────────────────────────────────────
# Shims FIRST so kernel headers resolve to HAL, then project root for runtime/
target_include_directories(hypertensor PRIVATE
    ${CMAKE_SOURCE_DIR}/host/shims
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/host
)

# ── Compile definitions ─────────────────────────────────────────────────────
target_compile_definitions(hypertensor PRIVATE HYPERTENSOR_HOSTED=1)

# ── Compiler flags ──────────────────────────────────────────────────────────
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hypertensor PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
        -Wno-sign-compare -Wno-missing-field-initializers
        -march=native
        -O2
    )
elseif(MSVC)
    target_compile_options(hypertensor PRIVATE
        /W3 /O2 /arch:AVX2
    )
endif()

# ── Platform libraries ──────────────────────────────────────────────────────
if(WIN32)
    target_link_libraries(hypertensor PRIVATE advapi32)
else()
    target_link_libraries(hypertensor PRIVATE pthread m)
endif()

# ── Install ─────────────────────────────────────────────────────────────────
install(TARGETS hypertensor RUNTIME DESTINATION bin)
