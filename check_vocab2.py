"""Read tokens from GGUF file for Gemma4 vocab check."""
import struct, sys

gguf_path = r"..\models\google_gemma-4-E2B-it-Q4_0.gguf"

with open(gguf_path, "rb") as f:
    # GGUF header
    magic = f.read(4)
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    
    def read_str(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(int(n)).decode("utf-8", errors="replace")
    
    def skip_value(f, vtype):
        if vtype == 8:   # string
            n = struct.unpack("<Q", f.read(8))[0]; f.seek(int(n), 1)
        elif vtype == 4: f.seek(4, 1)   # uint32
        elif vtype == 0: f.seek(1, 1)   # uint8
        elif vtype == 1: f.seek(1, 1)   # int8
        elif vtype == 2: f.seek(2, 1)   # uint16
        elif vtype == 3: f.seek(2, 1)   # int16
        elif vtype == 5: f.seek(4, 1)   # int32
        elif vtype == 6: f.seek(4, 1)   # float32
        elif vtype == 7: f.seek(1, 1)   # bool
        elif vtype == 9: f.seek(8, 1)   # uint64
        elif vtype == 10: f.seek(8, 1)  # int64
        elif vtype == 12: f.seek(8, 1)  # float64
        elif vtype == 11:                # array
            elem_type = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if elem_type == 8:  # string array - read only first 120 then skip
                for idx in range(int(n)):
                    slen = struct.unpack("<Q", f.read(8))[0]
                    s = f.read(int(slen))
                    if idx in (0, 1, 2, 104, 105, 106, 107, 108, 109, 256, 257):
                        text = s.decode("utf-8", errors="replace")
                        print(f"  tokens[{idx}] = {repr(text)}")
                return True
            elif elem_type == 4:   f.seek(int(n) * 4, 1)
            elif elem_type == 6:   f.seek(int(n) * 4, 1)
            elif elem_type == 5:   f.seek(int(n) * 4, 1)
            elif elem_type == 0:   f.seek(int(n), 1)
            elif elem_type == 7:   f.seek(int(n), 1)
            else:
                print(f"  Unknown array elem_type={elem_type}, n={n}")
        else:
            raise ValueError(f"Unknown vtype {vtype} at pos {f.tell()}")
        return False
    
    for _ in range(n_kv):
        key = read_str(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        if key == "tokenizer.ggml.tokens":
            print(f"Key: {key}")
            done = skip_value(f, vtype)
            break
        else:
            skip_value(f, vtype)
