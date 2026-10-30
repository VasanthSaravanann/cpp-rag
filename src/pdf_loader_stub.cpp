#include <stdexcept>
#include <string>

namespace cpp_rag {

std::string extract_pdf_text(const std::string&) {
    throw std::runtime_error(
        "PDF ingestion is unavailable on this platform; build with a PDF extraction backend");
}

}  // namespace cpp_rag
