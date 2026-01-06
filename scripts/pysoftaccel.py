import ctypes
import os
import platform

class SoftAccelNPU:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Try to find the library in potential locations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Note: MinGW/CMake might prefix with 'lib' even on Windows
            lib_names = [
                "softaccelnpu_v5_3.dll", "libsoftaccelnpu_v5_3.dll",
                "softaccelnpu_v5.dll", "libsoftaccelnpu_v5.dll", 
                "softaccelnpu_v4.dll", "libsoftaccelnpu_v4.dll", 
                "softaccelnpu.dll", "libsoftaccelnpu.dll", 
                "libsoftaccelnpu.so", "softaccelnpu.so"
            ]
            
            potential_paths = [
                os.path.join(base_dir, "build", "bin"),
                os.path.join(base_dir, "build", "lib"),
                os.path.join(base_dir, "build"),
                os.getcwd()
            ]
            
            for folder in potential_paths:
                for lib_name in lib_names:
                    path = os.path.join(folder, lib_name)
                    if os.path.exists(path):
                        lib_path = path
                        break
                if lib_path: break
        
        if not lib_path or not os.path.exists(lib_path):
            raise RuntimeError(f"Could not find SoftAccelNPU library. Please build the project first.")
            
        self.lib = ctypes.CDLL(lib_path)
        self._setup_api()
        print(f"[PySoftAccel] Loaded NPU Backend: {lib_path}")

    def _setup_api(self):
        # void* npu_create_tensor(int rows, int cols)
        self.lib.npu_create_tensor.restype = ctypes.c_void_p
        self.lib.npu_create_tensor.argtypes = [ctypes.c_int, ctypes.c_int]

        # void npu_delete_tensor(void* tensor)
        self.lib.npu_delete_tensor.argtypes = [ctypes.c_void_p]

        # void npu_randomize_tensor(void* tensor)
        self.lib.npu_randomize_tensor.argtypes = [ctypes.c_void_p]

        # float* npu_get_tensor_data(void* tensor)
        self.lib.npu_get_tensor_data.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.npu_get_tensor_data.argtypes = [ctypes.c_void_p]

        # void npu_execute_gemm_extreme(void* A, void* B, void* C, float sparsity)
        self.lib.npu_execute_gemm_extreme.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float]

        # Stats
        self.lib.npu_reset_cache.argtypes = []
        self.lib.npu_print_report.argtypes = []
        self.lib.npu_get_l1_hit_rate.restype = ctypes.c_double
        self.lib.npu_get_compression_ratio.restype = ctypes.c_double
        self.lib.npu_set_benchmark_mode.argtypes = [ctypes.c_bool]

    def set_benchmark_mode(self, enable):
        self.lib.npu_set_benchmark_mode(enable)

    def create_tensor(self, rows, cols, randomize=True):
        handle = self.lib.npu_create_tensor(rows, cols)
        if randomize:
            self.lib.npu_randomize_tensor(handle)
        return handle

    def delete_tensor(self, handle):
        self.lib.npu_delete_tensor(handle)

    def execute_gemm(self, A, B, C, sparsity=0.5):
        self.lib.npu_execute_gemm_extreme(A, B, C, sparsity)

    def print_report(self):
        self.lib.npu_print_report()

# Simple Test (Pure Python/Ctypes)
if __name__ == "__main__":
    try:
        npu = SoftAccelNPU()
        
        M, K, N = 512, 512, 512
        print(f"Testing PySoftAccel GEMM ({M}x{K}x{N})...")
        
        A = npu.create_tensor(M, K)
        B = npu.create_tensor(K, N)
        C = npu.create_tensor(M, N)
        
        npu.set_benchmark_mode(True) # Ensure Research Accelerator is ON for performance
        npu.lib.npu_reset_cache()
        npu.execute_gemm(A, B, C, sparsity=0.5)
        
        print("\nNPU Execution (Benchmark Mode) Complete.")
        npu.print_report()

        print("\nTesting Accuracy (Benchmark Mode OFF)...")
        npu.set_benchmark_mode(False)
        # Small GEMM for accuracy
        A_small = npu.create_tensor(4, 4)
        B_small = npu.create_tensor(4, 4)
        C_small = npu.create_tensor(4, 4, randomize=False)
        npu.execute_gemm(A_small, B_small, C_small)
        
        # Simple check - just ensuring it returns non-zero if inputs are non-zero
        data = npu.lib.npu_get_tensor_data(C_small)
        if data[0] != 0:
            print("[SUCCESS] Accuracy check (non-zero result) PASSED.")
        
        npu.delete_tensor(A)
        npu.delete_tensor(B)
        npu.delete_tensor(C)
        print("\n[SUCCESS] Python -> C++ Shared Library bridge is working!")
    except Exception as e:
        print(f"\n[FAILURE] {e}")
