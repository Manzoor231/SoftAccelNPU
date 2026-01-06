# SoftAccelNPU: 🚀 Developer & Engineering Usage Guide

This guide is designed to get you from "Hello World" to "Optimized LLM Inference" using the SoftAccelNPU library.

---

## 1. Quick Integration (The "User" Path)

To use SoftAccelNPU in your existing C++ project:

### Step A: Include and Setup

```cpp
#include "softaccelnpu/ops.h"
#include "softaccelnpu/hardware_info.h"

using namespace softaccelnpu;

int main() {
    // 1. Detect Hardware (Ryzen 5 3600 optimization)
    HardwareInfo::print_capabilities();
    GemmOps::tune_tiling(); // Auto-calibrates L1/L2/L3 blocks
```

### Step B: High-Performance GEMM

```cpp
    Tensor A(1024, 1024);
    Tensor B(1024, 1024);
    Tensor C(1024, 1024);
    
    A.randomize(); 
    B.randomize();
    C.fill(0.0f);

    // Runs a multi-threaded, tiled AVX2 execution
    GemmOps::gemm_tiled(A, B, C);
```

---

## 2. Developer Onboarding (The "Contributor" Path)

If you are a student or engineer looking to **modify** or **improve** the engine, here is where you work.

### 📍 Where to Add Code?

Reference the **[Project Structure Map](project_structure.md)** for full details.

1. **Adding a SIMD Kernel**: Edit `src/kernels/avx2_gemm.cpp`. Follow the `micro_kernel_6x16` pattern to add unrolled FMA logic.
2. **Experimenting with Tiling**: Modify `src/ops/gemm_tiled.cpp`. Change the `KC`, `MC`, or `NC` parameters in `GemmOps::tune_tiling()` to see how it affects performance on your specific CPU cache.
3. **Adding quantized precisions**: Look at `src/kernels/int4_avx2.cpp` to see how we handle nibble-packing and decompression.

### 🧪 Testing your changes

Always run the verification suite after any code modification:

```powershell
./bin/verify_accuracy.exe  # Ensures no numerical drift
./bin/demo_gemm.exe        # Measures performance regession
```

---

### 🤖 How to integrate with LLM Engines (GGML/llama.cpp)

If you are building an LLM runtime and want to replace standard CPU GEMM with **4D-V Acceleration**:

1. **Intercept the MatMul**: Find the `ggml_compute_forward_mul_mat` or equivalent function in your engine.
2. **Cast to SoftAccelNPU**:

    ```cpp
    // Example: Intercepting a 4096-dim layer
    void guest_engine_matmul(const float* A, const float* B, float* C) {
        softaccelnpu::Tensor tA(rows, cols, (void*)A); // Use external pointer
        softaccelnpu::Tensor tB(cols, k, (void*)B);
        softaccelnpu::Tensor tC(rows, k, (void*)C);
        
        // Unleash the 4D-V Cache
        softaccelnpu::GemmOps::gemm_tiled(tA, tB, tC);
    }
    ```

3. **Benchmark the Delta**: You should see the generation speed (tokens/sec) increase as the SoftAccelNPU skips zero-weights that standard CPU libraries like OpenBLAS might still compute.

### 🐍 Connecting to Python (AI Research)

SoftAccelNPU provides a shared library (`.dll` or `.so`) that connects directly to Python.

```python
from scripts.pysoftaccel import SoftAccelNPU

# Connect to the backend
npu = SoftAccelNPU()

# Run an 'Extreme' mode operation (INT4 + 50% Sparsity)
# This simulates the performance profile of a high-end LLM accelerator
npu.execute_gemm(A, B, C, sparsity=0.5)
```

### DirectML-like Deferred Execution

For high-performance graphics or inference pipelines, use the CommandList API:

```cpp
auto device = DmlDevice::create();
auto cmd_list = device->create_command_list();
auto op = device->create_gemm_operator(2048, 2048, 2048);

// Record multiple operations
cmd_list->record_gemm(op, A, B, C);
cmd_list->record_bias_add(C, bias, C);

// Execute all at once to minimize driver overhead
cmd_list->execute();
```

---

## 4. Requirements & Environment

* **Processor**: x86_64 with **AVX2 & FMA** support (Optimized for Ryzen 5 3600).
* **Compiler**: C++17 compliant (GCC 9+, MSVC 2019+).
* **OS**: Windows 10/11 or Ubuntu 20.04+.
* **Threads**: The library defaults to using all available logical cores (12 on Ryzen 3600).

---
*Created by the SoftAccelNPU Engineering Team.*
