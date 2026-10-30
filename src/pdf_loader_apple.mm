#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>

#include <stdexcept>
#include <string>

namespace cpp_rag {

std::string extract_pdf_text(const std::string& path) {
    @autoreleasepool {
        NSString* native_path = [[NSString alloc] initWithBytes:path.data()
                                                         length:path.size()
                                                       encoding:NSUTF8StringEncoding];
        if (native_path == nil) throw std::runtime_error("PDF path is not valid UTF-8");
        PDFDocument* document = [[PDFDocument alloc] initWithURL:[NSURL fileURLWithPath:native_path]];
        if (document == nil) throw std::runtime_error("unable to open PDF: " + path);

        std::string text;
        for (NSUInteger page_index = 0; page_index < document.pageCount; ++page_index) {
            PDFPage* page = [document pageAtIndex:page_index];
            NSString* page_text = page.string;
            if (page_text != nil) {
                const char* utf8 = page_text.UTF8String;
                if (utf8 != nullptr) text.append(utf8);
            }
            if (page_index + 1 < document.pageCount) text.append("\n\f\n");
        }
        return text;
    }
}

}  // namespace cpp_rag
