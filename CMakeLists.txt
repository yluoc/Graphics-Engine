cmake_minimum_required(VERSION 3.16)
project(MicrosecondGraphicsEngine CXX)

# ─────────────────────────────────────────────
# C++ Standard
# ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add these compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -ffast-math -funroll-loops -ftree-vectorize -march=native")

# ─────────────────────────────────────────────
# Aggressive Optimization Flags
# ─────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect best SIMD support
include(CheckCXXCompilerFlag)

set(SIMD_FLAGS)
set(OPT_FLAGS)

# Check for AVX-512
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    list(APPEND SIMD_FLAGS -mavx512f -mavx512dq -mavx512bw -mavx512vl)
    message(STATUS "SIMD: AVX-512 enabled")
else()
    # Check for AVX2 + FMA
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    if(COMPILER_SUPPORTS_AVX2)
        list(APPEND SIMD_FLAGS -mavx2)
        if(COMPILER_SUPPORTS_FMA)
            list(APPEND SIMD_FLAGS -mfma)
        endif()
        message(STATUS "SIMD: AVX2 + FMA enabled")
    else()
        # Check for SSE4.1
        check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
        if(COMPILER_SUPPORTS_SSE41)
            list(APPEND SIMD_FLAGS -msse4.1)
            message(STATUS "SIMD: SSE4.1 enabled")
        else()
            # Fallback to SSE2
            check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
            if(COMPILER_SUPPORTS_SSE2)
                list(APPEND SIMD_FLAGS -msse2)
                message(STATUS "SIMD: SSE2 enabled")
            else()
                message(STATUS "SIMD: Scalar fallback")
            endif()
        endif()
    endif()
endif()

# Aggressive optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND OPT_FLAGS
        -O3
        -DNDEBUG
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -fomit-frame-pointer
        -march=native
        -flto
    )
    
    # Link-time optimization
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ─────────────────────────────────────────────
# Include Directories
# ─────────────────────────────────────────────
set(PROJECT_DIR ${CMAKE_SOURCE_DIR}/source_code/graphics_engine)
set(INCLUDE_DIR ${PROJECT_DIR}/include)
set(SRC_DIR ${PROJECT_DIR}/src)
set(TEST_DIR ${PROJECT_DIR}/tests)

# ─────────────────────────────────────────────
# Optimized Engine Library
# ─────────────────────────────────────────────
set(ENGINE_SOURCES
    ${SRC_DIR}/simd_math.cpp
    ${SRC_DIR}/pipeline.cpp
    ${SRC_DIR}/vertex_manager.cpp
    ${SRC_DIR}/profiler.cpp
)

add_library(graphics_engine STATIC ${ENGINE_SOURCES})

target_include_directories(graphics_engine PUBLIC ${INCLUDE_DIR})
target_compile_options(graphics_engine PRIVATE ${SIMD_FLAGS} ${OPT_FLAGS})
target_link_libraries(graphics_engine pthread)

# ─────────────────────────────────────────────
# Benchmark Executable
# ─────────────────────────────────────────────
add_executable(benchmark
    ${TEST_DIR}/benchmark.cpp
)

target_include_directories(benchmark PRIVATE ${INCLUDE_DIR})
target_compile_options(benchmark PRIVATE ${SIMD_FLAGS} ${OPT_FLAGS})
target_link_libraries(benchmark graphics_engine pthread)

# ─────────────────────────────────────────────
# Test Executable
# ─────────────────────────────────────────────
enable_testing()

add_executable(test_engine
    ${TEST_DIR}/test_engine.cpp
)

target_include_directories(test_engine PRIVATE ${INCLUDE_DIR})
target_compile_options(test_engine PRIVATE ${SIMD_FLAGS} ${OPT_FLAGS})
target_link_libraries(test_engine graphics_engine pthread)

add_test(NAME EngineTests COMMAND test_engine)

# ─────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────
install(TARGETS graphics_engine ARCHIVE DESTINATION lib)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "  Microsecond Graphics Engine Configuration")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD flags:     ${SIMD_FLAGS}")
message(STATUS "  Optimization:   ${OPT_FLAGS}")
message(STATUS "═══════════════════════════════════════════════════")
message(STATUS "")
