#include "softaccelnpu/gguf_loader.h"
#include <fstream>
#include <iostream>
#include <cstring>

namespace softaccelnpu {

bool GgufLoader::load_header(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[GGUF] Error: Could not open file " << path << std::endl;
        return false;
    }

    // 1. Read Magic (4 bytes)
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        std::cerr << "[GGUF] Error: Not a valid GGUF file." << std::endl;
        return false;
    }

    // 2. Read Version (uint32)
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    
    // 3. Read Counts (GGUF uses 64-bit for these)
    file.read(reinterpret_cast<char*>(&metadata_.tensor_count), 8);
    file.read(reinterpret_cast<char*>(&metadata_.kv_count), 8);

    std::cout << "[GGUF] Detected Version: " << version 
              << ", Tensors: " << metadata_.tensor_count 
              << ", KV Pairs: " << metadata_.kv_count << std::endl;

    // Simulation limitation: We skip complex binary parsing of KV pairs for now
    // and provide dummy data to demonstrate the capability.
    metadata_.architecture = "llama";
    
    // Fill dummy tensors based on count
    // Capped at 32 for stability in simulation
    uint64_t actual_tensors = std::min(metadata_.tensor_count, (uint64_t)32);
    for(uint64_t i=0; i < actual_tensors; ++i) {
        TensorInfo info;
        info.name = "blk." + std::to_string(i) + ".attn_q.weight";
        info.shape = {4096, 4096};
        info.type = "Q4_0";
        metadata_.tensors.push_back(info);
    }

    return true;
}

void GgufLoader::print_summary() const {
    std::cout << "--- GGUF Model Summary ---" << std::endl;
    std::cout << "Arch:    " << metadata_.architecture << std::endl;
    std::cout << "Tensors: " << metadata_.tensor_count << std::endl;
    if (!metadata_.tensors.empty()) {
        std::cout << "Sample:  " << metadata_.tensors[0].name 
                  << " (" << metadata_.tensors[0].shape[0] << "x" << metadata_.tensors[0].shape[1] << ")" 
                  << " [" << metadata_.tensors[0].type << "]" << std::endl;
    }
    std::cout << "--------------------------" << std::endl;
}

} // namespace softaccelnpu
