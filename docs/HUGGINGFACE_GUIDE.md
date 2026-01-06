# 🤗 Hugging Face Integration Guide for SoftAccelNPU

This guide explains how to download a real AI model from Hugging Face and use it with the SoftAccelNPU v6.0 engine.

---

## Step 1: Find a GGUF Model

SoftAccelNPU works with the **GGUF** format (the industry standard for local LLMs). You can find thousands of these on Hugging Face.

1. Go to [Hugging Face](https://huggingface.co/models?search=gguf).
2. Search for models from users like **"bartowski"** or **"TheBloke"**.
3. Choose a small model for testing, such as **Llama-3.2-1B-Instruct-GGUF**.
4. Go to the **"Files and versions"** tab and download a specific quantization file (e.g., `Llama-3.2-1B-Instruct-Q4_K_M.gguf`).

---

## Step 2: Prepare the Model

Once you have the `.gguf` file downloaded:

1. Copy the file into your `build` folder (where the `.exe` files are).
2. Or simply remember the full path to the file.

---

## Step 3: Run the Production Test

Open your terminal in the project directory and run:

```powershell
# Navigate to the build folder
cd build

# Run the production loader with your real model file
.\bin\production_model_load.exe "C:\Path\To\Your\Model.gguf"
```

---

## What Happens During the Test?

When you run the command above, SoftAccelNPU performs these production steps:

1. **GGUF Header Parsing**: The engine reads the binary file to find how many layers the model has.
2. **Layer Dispatch**: It identifies the first 3 layers and prepares them for the virtual NPU.
3. **ECO-Mode Benchmark**: It runs a performance test on those layers using **4D-V Gating** and **Kernel Fusion** to show you the energy efficiency (`pJ/Token`).

---

## Why use GGUF?

By using GGUF, we ensure that SoftAccelNPU is not just a "toy" simulator, but a production-grade engine that can theoretically handle the weights of any modern LLM found on Hugging Face.

> [!TIP]
> Use a **Q4_K_M** or **Q4_0** quantization file. These are optimized for the exactly the kind of INT4 sparsity that our software NPU leverages for max performance!
