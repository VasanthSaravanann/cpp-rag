#include "cpp_rag/document_loader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace cpp_rag {

std::string extract_pdf_text(const std::string& path);

std::string load_document_text(const std::string& path) {
    const std::filesystem::path file_path(path);
    auto extension = file_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (extension == ".pdf") {
        auto text = extract_pdf_text(path);
        if (text.empty()) {
            throw std::runtime_error(
                "PDF contains no extractable text; scanned PDFs require OCR: " + path);
        }
        return text;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open input file: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace cpp_rag
