cmake_minimum_required(VERSION 3.16)
project(HighPerformanceGraphicsEngine CXX)

# ─────────────────────────────────────────────
# C++ Standard
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────────────────────────
# SIMD Flags (Linux / GCC / Clang)
# ─────────────────────────────────────────────
include(CheckCXXCompilerFlag)

set(SIMD_FLAGS)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    list(APPEND SIMD_FLAGS -mavx2)
    message(STATUS "SIMD: AVX2 enabled")
else()
    check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
    if(COMPILER_SUPPORTS_SSE41)
        list(APPEND SIMD_FLAGS -msse4.1)
        message(STATUS "SIMD: SSE4.1 enabled")
    else()
        check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
        if(COMPILER_SUPPORTS_SSE2)
            list(APPEND SIMD_FLAGS -msse2)
            message(STATUS "SIMD: SSE2 enabled")
        else()
            message(STATUS "SIMD: Scalar fallback (no SIMD flags)")
        endif()
    endif()
endif()

# ─────────────────────────────────────────────
# Optimization
# ─────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_Debug   "-g -O0")

# ─────────────────────────────────────────────
# Include Directories
# ─────────────────────────────────────────────
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR     ${CMAKE_SOURCE_DIR}/src)
set(TEST_DIR    ${CMAKE_SOURCE_DIR}/tests)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/../../examples)

# ─────────────────────────────────────────────
# Engine Library (static)
# ─────────────────────────────────────────────
add_library(graphics_engine STATIC
    ${SRC_DIR}/simd_math.cpp
    ${SRC_DIR}/vertex_manager.cpp
    ${SRC_DIR}/pipeline.cpp
    ${SRC_DIR}/profiler.cpp
)

target_include_directories(graphics_engine PUBLIC ${INCLUDE_DIR})
target_compile_options(graphics_engine PRIVATE ${SIMD_FLAGS})
target_link_libraries(graphics_engine pthread)

# ─────────────────────────────────────────────
# Test Executable
# ─────────────────────────────────────────────
add_executable(test_engine
    ${TEST_DIR}/test_engine.cpp
)

target_include_directories(test_engine PRIVATE ${INCLUDE_DIR})
target_compile_options(test_engine PRIVATE ${SIMD_FLAGS})
target_link_libraries(test_engine graphics_engine pthread)

# ─────────────────────────────────────────────
# Example Executable
# ─────────────────────────────────────────────
add_executable(market_simulator
    ${EXAMPLES_DIR}/market_simulator.cpp
)

target_include_directories(market_simulator PRIVATE ${INCLUDE_DIR})
target_compile_options(market_simulator PRIVATE ${SIMD_FLAGS})
target_link_libraries(market_simulator graphics_engine pthread)

# ─────────────────────────────────────────────
# CTest Integration
# ─────────────────────────────────────────────
enable_testing()
add_test(NAME EngineTests COMMAND test_engine)
