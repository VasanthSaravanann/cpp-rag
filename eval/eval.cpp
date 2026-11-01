// Retrieval/answer measurement harness. Runs the cpp-rag CLI as a subprocess so
// the numbers describe the shipped binary, not a re-implementation of it.
//
// The cases it reads are unverified scaffolding, not a labelled gold set, so the
// reported metrics are a regression signal and never a quality verdict. This
// program deliberately has no pass/fail thresholds.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {

struct Case {
    std::string id;
    std::string document;
    std::string question;
    std::string expect_substring;
    bool no_evidence{false};
};

struct Options {
    std::string bin{"build/cpp-rag"};
    std::string embed{"models/bge-small-en-v1.5-f16.gguf"};
    std::string generator{"models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"};
    std::string cases_path{"eval/cases.json"};
    std::string cache{"eval/.cache"};
    std::string scales{"1000:150,300:40"};
    std::size_t top_k{5};
    std::size_t per_scale{0};
    std::size_t max_tokens{96};
    std::size_t limit{0};
    std::string metric{"cosine"};
    bool generate{false};
    bool verbose{false};
    bool keep{false};
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open " + path.string());
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string lower(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) out.push_back(static_cast<char>(std::tolower(c)));
    return out;
}

bool contains(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return false;
    return lower(haystack).find(lower(needle)) != std::string::npos;
}

// --- Minimal JSON reader -------------------------------------------------
// Only the shapes cases.json uses are supported: objects, arrays, strings,
// booleans. No unicode escapes beyond \n \t \" \\ \/.
class Json {
public:
    enum class Kind { null, boolean, string, array, object };

    [[nodiscard]] Kind kind() const { return kind_; }
    [[nodiscard]] bool boolean() const { return boolean_; }
    [[nodiscard]] const std::string& string() const { return string_; }
    [[nodiscard]] const std::vector<Json>& items() const { return items_; }
    [[nodiscard]] const Json& at(const std::string& key) const {
        for (const auto& [k, v] : members_) {
            if (k == key) return v;
        }
        fail("missing json key: " + key);
    }
    [[nodiscard]] const Json* find(const std::string& key) const {
        for (const auto& [k, v] : members_) {
            if (k == key) return &v;
        }
        return nullptr;
    }

    static Json parse(std::string_view text) {
        Json out;
        std::size_t pos = 0;
        out = parse_value(text, pos);
        skip_ws(text, pos);
        if (pos != text.size()) fail("trailing content in json");
        return out;
    }

private:
    static void skip_ws(std::string_view t, std::size_t& p) {
        while (p < t.size() && std::isspace(static_cast<unsigned char>(t[p]))) ++p;
    }
    static void expect(std::string_view t, std::size_t& p, char c) {
        if (p >= t.size() || t[p] != c) {
            fail(std::string("expected '") + c + "' in json at offset " + std::to_string(p));
        }
        ++p;
    }
    static Json parse_value(std::string_view t, std::size_t& p) {
        skip_ws(t, p);
        if (p >= t.size()) fail("unexpected end of json");
        Json v;
        switch (t[p]) {
            case '{':
                ++p;
                skip_ws(t, p);
                v.kind_ = Kind::object;
                if (p < t.size() && t[p] == '}') { ++p; return v; }
                while (true) {
                    skip_ws(t, p);
                    auto key = parse_value(t, p);
                    skip_ws(t, p);
                    expect(t, p, ':');
                    v.members_.emplace_back(key.string_, parse_value(t, p));
                    skip_ws(t, p);
                    if (p < t.size() && t[p] == ',') { ++p; continue; }
                    expect(t, p, '}');
                    return v;
                }
            case '[':
                ++p;
                skip_ws(t, p);
                v.kind_ = Kind::array;
                if (p < t.size() && t[p] == ']') { ++p; return v; }
                while (true) {
                    v.items_.push_back(parse_value(t, p));
                    skip_ws(t, p);
                    if (p < t.size() && t[p] == ',') { ++p; continue; }
                    expect(t, p, ']');
                    return v;
                }
            case '"': {
                v.kind_ = Kind::string;
                ++p;
                while (p < t.size() && t[p] != '"') {
                    if (t[p] == '\\') {
                        ++p;
                        if (p >= t.size()) fail("bad escape in json string");
                        const char e = t[p++];
                        switch (e) {
                            case 'n': v.string_.push_back('\n'); break;
                            case 't': v.string_.push_back('\t'); break;
                            case 'r': v.string_.push_back('\r'); break;
                            case '"': v.string_.push_back('"'); break;
                            case '\\': v.string_.push_back('\\'); break;
                            case '/': v.string_.push_back('/'); break;
                            default: fail("unsupported json escape");
                        }
                    } else {
                        v.string_.push_back(t[p++]);
                    }
                }
                expect(t, p, '"');
                return v;
            }
            case 't':
                if (t.substr(p, 4) != "true") fail("bad json literal");
                p += 4;
                v.kind_ = Kind::boolean;
                v.boolean_ = true;
                return v;
            case 'f':
                if (t.substr(p, 5) != "false") fail("bad json literal");
                p += 5;
                v.kind_ = Kind::boolean;
                v.boolean_ = false;
                return v;
            case 'n':
                if (t.substr(p, 4) != "null") fail("bad json literal");
                p += 4;
                v.kind_ = Kind::null;
                return v;
            default:
                fail("unsupported json value at offset " + std::to_string(p));
        }
    }

    Kind kind_{Kind::null};
    bool boolean_{false};
    std::string string_;
    std::vector<Json> items_;
    std::vector<std::pair<std::string, Json>> members_;
};

// --- CLI invocation ------------------------------------------------------

std::string shell_quote(const std::string& value) {
    std::string out{"'"};
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}

struct Output {
    int status{};
    std::string text;
};

Output run_command(const std::vector<std::string>& argv, const fs::path& log_path,
                   bool verbose) {
    std::string command;
    for (const auto& arg : argv) command += shell_quote(arg) + " ";
    command += "2>&1";
    if (verbose) std::cerr << "+ " << command << '\n';
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) fail("popen failed");
    Output out;
    std::ofstream log(log_path);
    char buffer[4096];
    while (std::fgets(buffer, sizeof buffer, pipe) != nullptr) {
        out.text += buffer;
        log << buffer;
    }
    const int status = ::pclose(pipe);
    out.status = status == -1 ? -1 : WEXITSTATUS(status);
    if (out.status != 0) {
        std::cerr << "command failed (exit " << out.status << "), log: " << log_path << '\n'
                  << out.text.substr(out.text.size() > 4000 ? out.text.size() - 4000 : 0);
        fail("cpp-rag invocation failed");
    }
    return out;
}

// A query result is everything up to the next "score=" line or the end. Chunks
// whose text happens to contain that prefix are rare, so header-based splitting
// is good enough for this harness.
std::vector<std::string> parse_chunks(const std::string& text) {
    std::vector<std::string> chunks;
    // Skip loader/diagnostic lines emitted before the first result header.
    auto start = text.find("score=");
    if (start == std::string::npos) return chunks;
    if (start != 0 && text[start - 1] != '\n') start = std::string::npos;
    if (start == std::string::npos) return chunks;
    auto push = [&](std::size_t from, std::size_t to) {
        if (to > from) chunks.push_back(text.substr(from, to - from));
    };
    while (start < text.size()) {
        const auto next = text.find("\nscore=", start + 1);
        if (next == std::string::npos) {
            push(start, text.size());
            break;
        }
        push(start, next + 1);
        start = next + 1;
    }
    return chunks;
}

std::size_t arg_uint(const std::string& value) {
    return static_cast<std::size_t>(std::stoull(value));
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) fail("missing value for " + a);
            return argv[++i];
        };
        if (a == "--bin") o.bin = need();
        else if (a == "--embed") o.embed = need();
        else if (a == "--generator") o.generator = need();
        else if (a == "--cases") o.cases_path = need();
        else if (a == "--cache") o.cache = need();
        else if (a == "--scales") o.scales = need();
        else if (a == "--k") o.top_k = arg_uint(need());
        else if (a == "--per-scale") o.per_scale = arg_uint(need());
        else if (a == "--max-tokens") o.max_tokens = arg_uint(need());
        else if (a == "--limit") o.limit = arg_uint(need());
        else if (a == "--metric") o.metric = need();
        else if (a == "--generate") o.generate = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--keep") o.keep = true;
        else if (a == "--help") {
            std::cout << "usage: cpp_rag_eval [--scales S] [--k N] [--per-scale N] [--generate]\n"
                         "                   [--limit N] [--metric cosine|dot] [--keep] [--verbose]\n";
            std::exit(0);
        } else fail("unknown option: " + a);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto o = parse_options(argc, argv);
        const auto root = Json::parse(read_file(o.cases_path));
        const auto& provenance = root.at("provenance").string();
        std::cerr << "case provenance: " << provenance << "\n\n";

        fs::create_directories(o.cache);
        std::vector<std::string> index_argv{
            o.bin, "index", "--model", o.embed, "--output", o.cache + "/eval.ragidx"};
        for (const auto& d : root.at("documents").items()) {
            index_argv.push_back("--input");
            index_argv.push_back(d.string());
        }
        if (!o.scales.empty()) {
            index_argv.push_back("--scales");
            index_argv.push_back(o.scales);
        }
        const auto indexed = run_command(index_argv, o.cache + "/index.log", o.verbose);
        for (const auto& line : parse_chunks(indexed.text)) std::cerr << line;

        std::size_t hits = 0;
        double mrr = 0.0;
        std::size_t answered = 0;
        std::size_t answer_hits = 0;
        std::size_t abstain_total = 0;
        std::size_t abstain_ok = 0;
        std::size_t count = 0;

        std::cout << std::left << std::setw(18) << "case" << std::right << std::setw(4) << "hit"
                  << std::setw(8) << "rr" << std::setw(10) << "chunks";
        if (o.generate) std::cout << std::setw(9) << "answer";
        std::cout << '\n' << std::string(40 + (o.generate ? 9 : 0), '-') << '\n';

        for (const auto& c : root.at("cases").items()) {
            if (o.limit != 0 && count >= o.limit) break;
            Case item{c.at("id").string(),
                      c.at("document").string(),
                      c.at("question").string(),
                      c.at("expect_substring").string(),
                      c.at("no_evidence").boolean()};
            const auto base = o.cache + "/" + item.id;
            std::vector<std::string> query_argv{o.bin, "query", "--model", o.embed,
                                                "--index", o.cache + "/eval.ragidx",
                                                "--query", item.question,
                                                "--top-k", std::to_string(o.top_k),
                                                "--metric", o.metric};
            if (o.per_scale != 0) {
                query_argv.push_back("--per-scale");
                query_argv.push_back(std::to_string(o.per_scale));
            }
            const auto retrieved = parse_chunks(run_command(query_argv, base + ".query.log",
                                                            o.verbose).text);

            std::size_t first_hit = 0;
            for (std::size_t i = 0; i < retrieved.size(); ++i) {
                if (contains(retrieved[i], item.expect_substring)) {
                    first_hit = i + 1;
                    break;
                }
            }
            const double rr = first_hit == 0 ? 0.0 : 1.0 / static_cast<double>(first_hit);
            if (first_hit != 0) ++hits;
            mrr += rr;

            std::cout << std::left << std::setw(18) << item.id << std::right << std::setw(4)
                      << (first_hit != 0 ? "yes" : "no") << std::setw(8) << std::fixed
                      << std::setprecision(2) << rr << std::setw(10) << retrieved.size();

            if (o.generate) {
                std::vector<std::string> ask_argv{
                    o.bin, "ask", "--model", o.embed, "--generator", o.generator,
                    "--index", o.cache + "/eval.ragidx", "--query", item.question,
                    "--top-k", std::to_string(o.top_k), "--metric", o.metric,
                    "--max-tokens", std::to_string(o.max_tokens), "--ctx", "4096",
                    "--gpu-layers", "99"};
                if (o.per_scale != 0) {
                    ask_argv.push_back("--per-scale");
                    ask_argv.push_back(std::to_string(o.per_scale));
                }
                const auto answer = run_command(ask_argv, base + ".ask.log", o.verbose).text;
                const bool ok = contains(answer, item.expect_substring);
                ++answered;
                if (ok) ++answer_hits;
                if (item.no_evidence) {
                    ++abstain_total;
                    if (ok) ++abstain_ok;
                }
                std::cout << std::setw(9) << (ok ? "match" : "miss");
            }
            std::cout << '\n';
            ++count;
        }

        const double n = static_cast<double>(std::max<std::size_t>(count, 1));
        std::cout << "\nconfiguration: scales=" << (o.scales.empty() ? "(single)" : o.scales)
                  << " top-k=" << o.top_k << " per-scale=" << o.per_scale
                  << " metric=" << o.metric << '\n';
        std::cout << "cases      : " << count << '\n';
        std::cout << "hit@" << o.top_k << "    : " << std::fixed << std::setprecision(3)
                  << static_cast<double>(hits) / n << " (" << hits << '/' << count << ")\n";
        std::cout << "MRR        : " << mrr / n << '\n';
        if (o.generate) {
            std::cout << "answer match: " << std::setprecision(3)
                      << static_cast<double>(answer_hits) /
                             static_cast<double>(std::max<std::size_t>(answered, 1))
                      << " (" << answer_hits << '/' << answered << ")\n";
            std::cout << "abstain match: " << abstain_ok << '/' << abstain_total << '\n';
        }
        std::cout << "\nNote: unverified scaffolding cases. Not a quality verdict; no thresholds "
                     "are applied.\n";
        if (!o.keep) {
            std::error_code ignored;
            fs::remove_all(o.cache, ignored);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "eval error: " << e.what() << '\n';
        return 1;
    }
}
