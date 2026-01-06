import struct

def create_dummy_gguf(path):
    # Magic: 'GGUF'
    magic = b'GGUF'
    version = 3
    tensor_count = 10
    kv_count = 0
    
    with open(path, 'wb') as f:
        f.write(magic)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<Q', tensor_count))
        f.write(struct.pack('<Q', kv_count))
    print(f"Created dummy GGUF file at {path}")

if __name__ == "__main__":
    create_dummy_gguf("build/dummy_model.gguf")
