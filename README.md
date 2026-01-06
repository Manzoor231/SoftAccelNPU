⚡ SoftAccelNPU
Software-Defined 4D-V Neural Processing Engine

SoftAccelNPU is a high-performance, open-source AI acceleration engine that brings NPU-class inference performance to standard x86 CPUs.
It introduces the proprietary 4D-V Software-Defined Cache Technology, optimized and benchmarked on AMD Ryzen™ 5 3600.

<p align="left"> <a href="https://opensource.org/licenses/MIT"> <img src="https://img.shields.io/badge/License-MIT-yellow.svg" /> </a> <a href="docs/benchmarks.md"> <img src="https://img.shields.io/badge/Precision-INT4%20%7C%20INT8%20%7C%20FP32-blue" /> </a> <a href="docs/architecture.md"> <img src="https://img.shields.io/badge/Technology-4D--V%20Cache-blueviolet" /> </a> </p>
💎 The 4D-V Breakthrough

SoftAccelNPU debuts the 4D-V Cache, a novel software-defined execution model that applies value-pattern awareness to eliminate redundant compute paths at runtime.

On a mid-range Ryzen 5 3600, the engine reaches an effective 111.45 TOPS, rivaling integrated mobile NPUs such as those found in Apple Silicon—without dedicated AI hardware.

🚀 Interactive NPU Profiler (Standalone)

New in v5.2

SoftAccelNPU now includes a fully standalone interactive profiler—no terminal, no VS Code, no toolchain friction.

🧠 Real-time telemetry and throughput metrics

📊 Benchmark selection via UI

⚙️ Zero-configuration startup

👉 Launch:
Run run_profiler.bat from the repository root.

🖥️ Performance Snapshot

Target Platform: AMD Ryzen™ 5 3600

Mode	Throughput	Capability
FP32	294.09 GFLOPS	High-precision scientific & research workloads
INT8	8.22 TOPS	Mobile-class quantized inference
4D-V INT4	111.45 TOPS (Effective)	Large-Language-Model acceleration (LLaMA-2 class)
🛠️ Architecture Overview

SoftAccelNPU combines a DirectML-style frontend API with low-level AVX2 SIMD execution, forming a vertically optimized software NPU stack.

Core Layers:

Frontend

C++ API

Python bridge (scripts/pysoftaccel.py)

Scheduler

Hierarchical tiling & dispatch engine

(src/ops/gemm_tiled.cpp)

Execution Engine

4D-V value-aware cache

Hand-optimized SIMD kernels

(src/kernels/)

📂 Repository Guide
Path	Description
include/softaccelnpu/	Public C/C++ headers
docs/	Architecture deep-dives and project structure
examples/	Standalone profiler and validation tests

Key documentation:

📐 Architecture

🧱 Project Structure

📦 One-Click Startup

No IDE Required

Ensure the build/ directory is ready
→ See Usage Guide

Double-click run_profiler.bat

Select benchmarks and inspect live telemetry

🌍 Vision

SoftAccelNPU is designed as a research-grade, production-capable software NPU, enabling:

CPU-only AI acceleration

Edge and low-power inference

Experimental AI runtime research

<p align="center"> <em>Created for the Open-Source Community.</em><br/> <strong>© 2026 · Manzoor Strange</strong> </p>

