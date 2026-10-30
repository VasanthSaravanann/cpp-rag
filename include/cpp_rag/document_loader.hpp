#pragma once

#include <string>

namespace cpp_rag {

// Loads UTF-8 text files directly and extracts page text from PDF files.
[[nodiscard]] std::string load_document_text(const std::string& path);

}  // namespace cpp_rag
