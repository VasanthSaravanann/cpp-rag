#include "cpp_rag/mmap_index.hpp"

#include <algorithm>
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
constexpr std::uint32_t kVersion = 3;
// Written in host byte order. A reader on the opposite byte order sees the swapped
// value and can name the problem instead of reporting a bogus version.
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kChecksumFnv1a64 = 1U;

constexpr std::uint32_t byte_swap32(std::uint32_t value) {
    return ((value & 0x000000FFU) << 24) | ((value & 0x0000FF00U) << 8) |
           ((value & 0x00FF0000U) >> 8) | ((value & 0xFF000000U) >> 24);
}

struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint32_t endian_marker;
    std::uint32_t record_size;
    std::uint64_t item_count;
    std::uint32_t dimensions;
    std::uint32_t flags;
    std::uint64_t records_offset;
    std::uint64_t vectors_offset;
    std::uint64_t strings_offset;
    std::uint64_t file_size;
    std::uint32_t checksum_algorithm;
    std::uint32_t reserved;
    std::uint64_t payload_checksum;
    std::uint64_t reserved2;
};
static_assert(sizeof(Header) == 96);

// FNV-1a over the bytes after the header. It is not cryptographic: it detects
// accidental corruption such as bit rot or a truncated write, not tampering.
struct Fnv1a64 {
    std::uint64_t state{14695981039346656037ULL};

    void add(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            state ^= bytes[i];
            state *= 1099511628211ULL;
        }
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return state; }

    [[nodiscard]] static std::uint64_t of(const void* data, std::size_t size) {
        Fnv1a64 hasher;
        hasher.add(data, size);
        return hasher.value();
    }
};

struct Record {
    std::uint64_t source_offset;
    std::uint64_t source_size;
    std::uint64_t text_offset;
    std::uint64_t text_size;
    std::uint64_t byte_begin;
    std::uint64_t byte_end;
    std::uint64_t vector_index;
    float norm;
    std::uint32_t scale;
    std::uint32_t scale_bytes;
    std::uint32_t reserved;
};
static_assert(sizeof(Record) == 72);

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void write_padding(std::ofstream& output, Fnv1a64& hasher, std::uint64_t bytes) {
    static constexpr char zeros[64]{};
    while (bytes > 0) {
        const auto count = static_cast<std::streamsize>(std::min<std::uint64_t>(bytes, sizeof(zeros)));
        output.write(zeros, count);
        hasher.add(zeros, static_cast<std::size_t>(count));
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
    file_header.endian_marker = kEndianMarker;
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
        item_record.scale = item.chunk.scale;
        item_record.scale_bytes = item.chunk.scale_bytes == 0
            ? static_cast<std::uint32_t>(std::max<std::size_t>(1, item.chunk.text.size()))
            : item.chunk.scale_bytes;
        double squared_norm = 0.0;
        for (const auto value : item.embedding) {
            squared_norm += static_cast<double>(value) * value;
        }
        item_record.norm = static_cast<float>(std::sqrt(squared_norm));
    }
    file_header.file_size = string_cursor;
    file_header.checksum_algorithm = kChecksumFnv1a64;

    const auto temporary = path + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("unable to create index: " + temporary);
    }
    // The checksum covers everything after the header, so the header is written
    // twice: a placeholder now, then again with the digest once the payload is out.
    output.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));

    Fnv1a64 hasher;
    const auto emit = [&](const void* data, std::size_t size) {
        output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        hasher.add(data, size);
    };
    emit(records.data(), records.size() * sizeof(Record));
    write_padding(output, hasher,
                  file_header.vectors_offset -
                      (file_header.records_offset + records.size() * sizeof(Record)));
    for (const auto& item : items) {
        emit(item.embedding.data(), item.embedding.size() * sizeof(float));
    }
    for (const auto& item : items) {
        emit(item.chunk.source.data(), item.chunk.source.size());
        emit(item.chunk.text.data(), item.chunk.text.size());
    }
    file_header.payload_checksum = hasher.value();
    output.seekp(0);
    output.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
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
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("invalid index magic");
    }
    // Version first, so a v1/v2 file reports its real problem: in those layouts the
    // bytes at the endian-marker offset were the record size. A v3 file from the
    // opposite byte order reads as an unknown version here, but its marker gives the
    // specific reason.
    if (h.version != kVersion) {
        if (h.endian_marker == byte_swap32(kEndianMarker)) {
            throw std::runtime_error(
                "index was written on a host with the opposite byte order; rebuild the index");
        }
        throw std::runtime_error("unsupported index version; rebuild the index for CPPRAGIX v3");
    }
    if (h.endian_marker != kEndianMarker) {
        if (h.endian_marker == byte_swap32(kEndianMarker)) {
            throw std::runtime_error(
                "index was written on a host with the opposite byte order; rebuild the index");
        }
        throw std::runtime_error("invalid index endianness marker");
    }
    if (h.header_size != sizeof(Header) || h.record_size != sizeof(Record) ||
        h.file_size != mapped_size_ || h.dimensions == 0 || h.item_count == 0 ||
        h.checksum_algorithm != kChecksumFnv1a64 || h.flags != 0) {
        throw std::runtime_error("invalid or unsupported index header");
    }

    // Section sizes are checked by division, never by multiplying a header field:
    // a crafted item_count or dimensions would otherwise wrap the byte-size
    // multiplication, pass range_valid(), and let the record loop read past the
    // mapping. Each check below also proves the multiplication that follows it.
    if (h.records_offset < sizeof(Header) || h.records_offset > h.file_size ||
        h.item_count > (h.file_size - h.records_offset) / sizeof(Record)) {
        throw std::runtime_error("index record section is invalid");
    }
    const auto records_end = h.records_offset + h.item_count * sizeof(Record);
    if (h.vectors_offset < records_end || h.vectors_offset > h.file_size ||
        h.vectors_offset % alignof(float) != 0 ||
        h.dimensions > (h.file_size - h.vectors_offset) / sizeof(float) / h.item_count) {
        throw std::runtime_error("index vector section is invalid");
    }
    const auto vector_end = h.vectors_offset + h.item_count * h.dimensions * sizeof(float);
    if (h.strings_offset < vector_end || h.strings_offset > h.file_size) {
        throw std::runtime_error("index string section is invalid");
    }

    for (std::size_t i = 0; i < h.item_count; ++i) {
        const auto& r = record(data_, i);
        if (!range_valid(r.source_offset, r.source_size, h.file_size) ||
            !range_valid(r.text_offset, r.text_size, h.file_size) ||
            r.source_offset < h.strings_offset ||
            r.text_offset < r.source_offset + r.source_size ||
            r.text_offset + r.text_size < r.text_offset ||
            r.vector_index >= h.item_count || r.byte_begin > r.byte_end ||
            r.scale_bytes == 0) {
            throw std::runtime_error("index record is invalid");
        }
    }

    // Verified last so structural errors keep their specific messages; anything the
    // section and record checks cannot see (flipped text/vector bytes) lands here.
    const auto checksum = Fnv1a64::of(data_ + sizeof(Header), mapped_size_ - sizeof(Header));
    if (checksum != h.payload_checksum) {
        throw std::runtime_error(
            "index payload checksum mismatch; the file is corrupt, rebuild the index");
    }
}

const std::byte* MmapIndex::bytes() const {
    if (data_ == nullptr) throw std::logic_error("index is not open");
    return data_;
}

std::size_t MmapIndex::size() const {
    return static_cast<std::size_t>(header(bytes()).item_count);
}

std::size_t MmapIndex::dimensions() const { return header(bytes()).dimensions; }

Chunk MmapIndex::chunk(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    const auto* data = bytes();
    const auto& r = record(data, index);
    const auto* chars = reinterpret_cast<const char*>(data);
    return {std::string(chars + r.source_offset, r.source_size),
            std::string(chars + r.text_offset, r.text_size),
            static_cast<std::size_t>(r.byte_begin), static_cast<std::size_t>(r.byte_end),
            r.scale, r.scale_bytes};
}

std::span<const float> MmapIndex::embedding(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    const auto* data = bytes();
    const auto vector_index = record(data, index).vector_index;
    // validate() accepted the whole vector section inside the file and aligned it to
    // float, so this offset cannot wrap and the cast is well aligned.
    const auto offset = header(data).vectors_offset + vector_index * dimensions() * sizeof(float);
    return {reinterpret_cast<const float*>(data + offset), dimensions()};
}

float MmapIndex::norm(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    return record(bytes(), index).norm;
}

std::uint32_t MmapIndex::scale(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    return record(bytes(), index).scale;
}

std::uint32_t MmapIndex::scale_bytes(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("index item out of range");
    return record(bytes(), index).scale_bytes;
}

}  // namespace cpp_rag
