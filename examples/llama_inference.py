import sys
import os

# Ensure we can find the scripts/pysoftaccel module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pysoftaccel import SoftAccelNPU
import time

def run_llama_inference():
    print("================================================================")
    print("   SoftAccelNPU: Llama-2-7B Python Inference Simulator")
    print("================================================================")
    
    try:
        npu = SoftAccelNPU()
        
        # Llama-2-7B FFN Dimensions
        DIM = 4096
        HIDDEN_DIM = 11008
        
        print(f"\n[Model] Llama-2-7B FFN Block")
        print(f"[Config] Dim={DIM}, Hidden={HIDDEN_DIM}")
        
        # Prepare Tensors
        # In a real FFN, we have: gate_proj, up_proj, down_proj
        # We simulate the most compute-intensive part
        print("[Memory] Allocating NPU Tensors...")
        x = npu.create_tensor(DIM, 1)
        w_gate = npu.create_tensor(HIDDEN_DIM, DIM)
        output = npu.create_tensor(HIDDEN_DIM, 1)
        
        print("\n[Benchmark] Running 100 simulated tokens...")
        npu.lib.npu_reset_cache()
        
        start_time = time.time()
        for i in range(100):
            # Simulate FFN Gate Projection with 50% Sparsity
            npu.execute_gemm(w_gate, x, output, sparsity=0.5)
            
        end_time = time.time()
        
        total_time = end_time - start_time
        tps = 100 / total_time
        
        print("\n----------------------------------------------------------------")
        print("   RESULTS (Python API)")
        print("----------------------------------------------------------------")
        print(f"Total Time:       {total_time:.4f} s")
        print(f"Throughput:       {tps:.2f} tokens/sec (Simulated)")
        
        npu.print_report()
        
        # Cleanup
        npu.delete_tensor(x)
        npu.delete_tensor(w_gate)
        npu.delete_tensor(output)
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    run_llama_inference()
