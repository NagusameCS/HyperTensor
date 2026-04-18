"""Find what tok=106 and tok=107 decode to using llm's vocab data"""
# Since we can't easily run the model just for vocab, let's look at the gguf file header
# and find the token pieces from the binary data
import struct, sys

gguf_path = r"..\models\google_gemma-4-E2B-it-Q4_0.gguf"
with open(gguf_path, "rb") as f:
    # GGUF magic
    magic = f.read(4)
    if magic != b"GGUF":
        print("Not a GGUF file!")
        sys.exit(1)
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} KV pairs")
    
    # Read KV pairs to find tokenizer.ggml.tokens
    def read_string(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")
    
    def read_value(f, vtype):
        if vtype == 8:  # string
            return read_string(f)
        elif vtype == 4:  # uint32
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == 11:  # array
            elem_type = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:  # array of strings
                return [read_string(f) for _ in range(n)]
            else:
                return f"array<{elem_type}>[{n}] (skipped)"
        elif vtype == 0:  # uint8
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == 1:  # int8
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == 2:  # uint16
            return struct.unpack("<H", f.read(2))[0]
        elif vtype == 3:  # int16
            return struct.unpack("<h", f.read(2))[0]
        elif vtype == 5:  # int32
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == 6:  # float32
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == 7:  # bool
            return bool(struct.unpack("<B", f.read(1))[0])
        elif vtype == 9:  # uint64
            return struct.unpack("<Q", f.read(8))[0]
        elif vtype == 10:  # int64
            return struct.unpack("<q", f.read(8))[0]
        elif vtype == 12:  # float64
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unknown vtype {vtype}")
    
    tokens = None
    for _ in range(n_kv):
        key = read_string(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        val = read_value(f, vtype)
        if key == "tokenizer.ggml.tokens":
            tokens = val
            print(f"Found tokens list: {len(tokens)} tokens")
            print(f"tok[106] = {repr(tokens[106])}")
            print(f"tok[107] = {repr(tokens[107])}")
            print(f"tok[108] = {repr(tokens[108])}")
            print(f"tok[1] = {repr(tokens[1])}")
            print(f"tok[2] = {repr(tokens[2])}")
            break
        elif key.startswith("tokenizer.ggml"):
            print(f"  {key}: {str(val)[:80]}")
