#pragma once

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace cpp_rag::detail {

// llama.cpp's backend is process-global, so it must outlive every model and context
// built on top of it. Instances keep a reference instead of initialising the backend
// themselves; the last guard to be destroyed frees it. Declare the guard before the
// model and context members so it is destroyed after them.
class LlamaBackendGuard {
public:
    LlamaBackendGuard() {
        auto& state = shared();
        const std::lock_guard<std::mutex> lock(state.mutex);
        if (state.references == 0) {
            // Model-load and scheduling chatter would drown out the CLI's own output.
            // Warnings and errors still print; CPP_RAG_VERBOSE=1 restores everything.
            if (std::getenv("CPP_RAG_VERBOSE") == nullptr) llama_log_set(quiet_log, nullptr);
            llama_backend_init();
        }
        ++state.references;
    }

    ~LlamaBackendGuard() {
        auto& state = shared();
        const std::lock_guard<std::mutex> lock(state.mutex);
        if (--state.references == 0) llama_backend_free();
    }

    LlamaBackendGuard(const LlamaBackendGuard&) = delete;
    LlamaBackendGuard& operator=(const LlamaBackendGuard&) = delete;
    LlamaBackendGuard(LlamaBackendGuard&&) = delete;
    LlamaBackendGuard& operator=(LlamaBackendGuard&&) = delete;

private:
    static void quiet_log(ggml_log_level level, const char* text, void* /*user_data*/) {
        if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO ||
            level == GGML_LOG_LEVEL_CONT) {
            return;
        }
        std::fputs(text, stderr);
    }

    struct State {
        std::mutex mutex;
        int references{};
    };

    // Shared by every translation unit: inline functions have one instance each.
    static State& shared() {
        static State state;
        return state;
    }
};

}  // namespace cpp_rag::detail
