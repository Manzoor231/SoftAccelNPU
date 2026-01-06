#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace softaccelnpu {

/**
 * @brief Lightweight GGUF (GPT-Generated Unified Format) Loader.
 * 
 * This class handles the parsing of GGUF model files to extract
 * tensor shapes, quantization types, and model metadata.
 */
class GgufLoader {
public:
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        std::string type; // e.g., "F32", "Q4_0", "Q8_0"
        uint64_t offset;
    };

    struct ModelMetadata {
        std::string architecture;
        uint64_t tensor_count;
        uint64_t kv_count;
        std::map<std::string, std::string> kv_pairs;
        std::vector<TensorInfo> tensors;
    };

    GgufLoader() = default;

    /**
     * @brief Parses a GGUF file header and extracts metadata.
     * Note: This is a lightweight reader for research and simulation.
     */
    bool load_header(const std::string& path);

    const ModelMetadata& get_metadata() const { return metadata_; }
    void print_summary() const;

private:
    ModelMetadata metadata_;
};

} // namespace softaccelnpu
