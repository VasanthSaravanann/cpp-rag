#include "cpp_rag/mmap_index.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cpp_rag {
namespace {

constexpr char kMagic[8] = {'C', 'P', 'P', 'R', 'A', 'G', 'I', 'X'};
constexpr std::uint32_t kVersion = 1;

struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint64_t item_count;
    std::uint32_t dimensions;
    std::uint32_t record_size;
    std::uint64_t records_offset;
    std::uint64_t vectors_offset;
    std::uint64_t strings_offset;
    std::uint64_t file_size;
};
static_assert(sizeof(Header) == 64);

struct Record {
    std::uint64_t source_offset;
    std::uint64_t source_size;
    std::uint64_t text_offset;
    std::uint64_t text_size;
    std::uint64_t byte_begin;
    std::uint64_t byte_end;
    std::uint64_t vector_index;
    float norm;
    std::uint32_t reserved;
};
static_assert(sizeof(Record) == 64);

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void write_padding(std::ofstream& output, std::uint64_t bytes) {
    static constexpr char zeros[64]{};
    while (bytes > 0) {
        const auto count = static_cast<std::streamsize>(std::min<std::uint64_t>(bytes, sizeof(zeros)));
        output.write(zeros, count);
        bytes -= static_cast<std::uint64_t>(count);
    }
}

const Header& header(const std::byte* data) {
    return *reinterpret_cast<const Header*>(data);
}

const Record& record(const std::byte* data, std::size_t index) {
    const auto& file_header = header(data);
    return *reinterpret_cast<const Record*>(data + file_header.records_offset + index * sizeof(Record));
}

bool range_valid(std::uint64_t offset, std::uint64_t size, std::uint64_t limit) {
    return offset <= limit && size <= limit - offset;
}

}  // namespace

void MmapIndex::write(const std::string& path, std::span<const IndexItem> items) {
    if (items.empty()) {
        throw std::invalid_argument("cannot write an empty index");
    }
    const auto dimensions = items.front().embedding.size();
    if (dimensions == 0 || dimensions > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid embedding dimensions");
    }
    for (const auto& item : items) {
        if (item.embedding.size() != dimensions) {
            throw std::invalid_argument("all embeddings must have equal dimensions");
        }
    }

    Header file_header{};
    std::memcpy(file_header.magic, kMagic, sizeof(kMagic));
    file_header.version = kVersion;
    file_header.header_size = sizeof(Header);
    file_header.item_count = items.size();
    file_header.dimensions = static_cast<std::uint32_t>(dimensions);
    file_header.record_size = sizeof(Record);
    file_header.records_offset = sizeof(Header);
    file_header.vectors_offset = align_up(file_header.records_offset + items.size() * sizeof(Record), 64);
    const auto vector_bytes = items.size() * dimensions * sizeof(float);
    file_header.strings_offset = file_header.vectors_offset + vector_bytes;

    std::vector<Record> records(items.size());
    std::uint64_t string_cursor = file_header.strings_offset;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        auto& item_record = records[i];
        item_record.source_offset = string_cursor;
        item_record.source_size = item.chunk.source.size();
        string_cursor += item_record.source_size;
        item_record.text_offset = string_cursor;
        item_record.text_size = item.chunk.text.size();
        string_cursor += item_record.text_size;
        item_record.byte_begin = item.chunk.byte_begin;
        item_record.byte_end = item.chunk.byte_end;
        item_record.vector_index = i;
        double squared_norm = 0.0;
        for (const auto value : item.embedding) {
            squared_norm += static_cast<double>(value) * value;
        }
        item_record.norm = static_cast<float>(std::sqrt(squared_norm));
    }
    file_header.file_size = string_cursor;

    const auto temporary = path + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create index: " + temporary);
    }
    output.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    output.write(reinterpret_cast<const char*>(records.data()),
                 static_cast<std::streamsize>(records.size() * sizeof(Record)));
    write_padding(output, file_header.vectors_offset -
                          (file_header.records_offset + records.size() * sizeof(Record)));
    for (const auto& item : items) {
        output.write(reinterpret_cast<const char*>(item.embedding.data()),
                     static_cast<std::streamsize>(item.embedding.size() * sizeof(float)));
    }
    for (const auto& item : items) {
        output.write(item.chunk.source.data(), static_cast<std::streamsize>(item.chunk.source.size()));
        output.write(item.chunk.text.data(), static_cast<std::streamsize>(item.chunk.text.size()));
    }
    output.close();
    if (!output) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("failed while writing index: " + temporary);
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        std::filesystem::remove(temporary);
        throw std::system_error(error, "unable to install index");
    }
}

MmapIndex::MmapIndex(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("unable to open index: " + path);
    }
    struct stat status {};
    if (::fstat(fd_, &status) != 0 || status.st_size <= 0) {
        close();
        throw std::runtime_error("unable to stat index: " + path);
    }
    mapped_size_ = static_cast<std::size_t>(status.st_size);
    const auto* mapping = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping == MAP_FAILED) {
        data_ = nullptr;
        close();
        throw std::runtime_error("unable to memory-map index: " + path);
    }
    data_ = static_cast<const std::byte*>(mapping);
    try {
        validate();
    } catch (...) {
        close();
        throw;
    }
}

MmapIndex::~MmapIndex() { close(); }

MmapIndex::MmapIndex(MmapIndex&& other) noexcept { *this = std::move(other); }

MmapIndex& MmapIndex::operator=(MmapIndex&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = std::exchange(other.fd_, -1);
        data_ = std::exchange(other.data_, nullptr);
        mapped_size_ = std::exchange(other.mapped_size_, 0);
    }
    return *this;
}

void MmapIndex::close() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), mapped_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = -1;
    data_ = nullptr;
    mapped_size_ = 0;
}

void MmapIndex::validate() {
    if (mapped_size_ < sizeof(Header)) {
        throw std::runtime_error("index header is truncated");
    }
    const auto& h = header(data_);
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0 || h.version != kVersion ||
        h.header_size != sizeof(Header) || h.record_size != sizeof(Record) ||
        h.file_size != mapped_size_ || h.dimensions == 0 || h.item_count == 0) {
        throw std::runtime_error("invalid or unsupported index header");
    }
    if (!range_valid(h.records_offset, h.item_count * sizeof(Record), h.file_size) ||
        !range_valid(h.vectors_offset, h.item_count * h.dimensions * sizeof(float), h.file_size) ||
        h.strings_offset < h.vectors_offset) {
        throw std::runtime_error("index data range is invalid");
    }
    for (std::size_t i = 0; i < h.item_count; ++i) {
        const auto& r = record(data_, i);
        if (!range_valid(r.source_offset, r.source_size, h.file_size) ||
            !range_valid(r.text_offset, r.text_size, h.file_size) ||
            r.source_offset < h.strings_offset ||
            r.text_offset < r.source_offset + r.source_size ||
            r.text_offset + r.text_size < r.text_offset ||
            r.vector_index >= h.item_count || r.byte_begin > r.byte_end) {
            throw std::runtime_error("index record is invalid");
        }
    }
}

std::size_t MmapIndex::size() const noexcept { return static_cast<std::size_t>(header(data_).item_count); }
std::size_t MmapIndex::dimensions() const noexcept { return header(data_).dimensions; }

Chunk MmapIndex::chunk(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    const auto& r = record(data_, index);
    const auto* chars = reinterpret_cast<const char*>(data_);
    return {std::string(chars + r.source_offset, r.source_size),
            std::string(chars + r.text_offset, r.text_size),
            static_cast<std::size_t>(r.byte_begin), static_cast<std::size_t>(r.byte_end)};
}

std::span<const float> MmapIndex::embedding(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    const auto vector_index = record(data_, index).vector_index;
    const auto offset = header(data_).vectors_offset + vector_index * dimensions() * sizeof(float);
    return {reinterpret_cast<const float*>(data_ + offset), dimensions()};
}

float MmapIndex::norm(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    return record(data_, index).norm;
}

}  // namespace cpp_rag
