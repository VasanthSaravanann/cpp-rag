cmake_minimum_required(VERSION 3.20)
project(cpp_rag VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPP_RAG_WITH_LLAMA "Build llama.cpp embedding integration" ON)
option(CPP_RAG_BUILD_TESTS "Build unit tests" ON)
set(LLAMA_CPP_DIR "" CACHE PATH "Path to an existing llama.cpp checkout")

add_library(cpp_rag_core STATIC
    src/chunker.cpp
    src/mmap_index.cpp
    src/prompt_builder.cpp
    src/vector_search.cpp
)
target_include_directories(cpp_rag_core PUBLIC include)
target_compile_features(cpp_rag_core PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(cpp_rag_core PRIVATE /W4)
else()
    target_compile_options(cpp_rag_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CPP_RAG_WITH_LLAMA)
    if(NOT TARGET llama)
        set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
        # AppleClang can hang while llama.cpp try-runs unsupported SVE/SME probes.
        if(APPLE AND NOT DEFINED GGML_NATIVE)
            set(GGML_NATIVE OFF CACHE BOOL "Disable problematic native feature probes")
        endif()

        if(LLAMA_CPP_DIR)
            add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")
        else()
            include(FetchContent)
            FetchContent_Declare(llama_cpp
                URL https://github.com/ggml-org/llama.cpp/archive/e21cdc11.tar.gz
                URL_HASH SHA256=b9473c1cd1acf3c5e80feb990566be399587ee1ee940c91b4278ca6eb146ff19
                DOWNLOAD_EXTRACT_TIMESTAMP FALSE
            )
            FetchContent_MakeAvailable(llama_cpp)
        endif()
    endif()

    add_library(cpp_rag_llama STATIC
        src/llama_embedder.cpp
        src/llama_generator.cpp
    )
    target_include_directories(cpp_rag_llama PUBLIC include)
    target_link_libraries(cpp_rag_llama PUBLIC cpp_rag_core llama)
    target_compile_definitions(cpp_rag_llama PUBLIC CPP_RAG_WITH_LLAMA=1)
endif()

add_executable(cpp-rag src/main.cpp)
if(CPP_RAG_WITH_LLAMA)
    target_link_libraries(cpp-rag PRIVATE cpp_rag_llama)
    target_compile_definitions(cpp-rag PRIVATE CPP_RAG_WITH_LLAMA=1)
else()
    target_link_libraries(cpp-rag PRIVATE cpp_rag_core)
endif()

if(CPP_RAG_BUILD_TESTS)
    enable_testing()
    add_executable(cpp_rag_tests tests/tests.cpp)
    target_link_libraries(cpp_rag_tests PRIVATE cpp_rag_core)
    add_test(NAME cpp_rag_tests COMMAND cpp_rag_tests)
endif()
