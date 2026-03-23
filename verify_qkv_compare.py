"""
Compute Q projection for token 32010 at L0 and compare against TensorOS.
TensorOS says: Q[0..3] = -0.0141, 0.0024, -0.0243, -0.0079
               Q_L2 (sum of sq) = 2.4324
"""
import numpy as np
import struct

GGUF_PATH = r'C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf'
dim = 3072
token_id = 32010

f = open(GGUF_PATH, 'rb')
magic = f.read(4); version = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]; n_kv = struct.unpack('<Q', f.read(8))[0]

def read_string():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(vtype):
    if vtype == 0: return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7: return struct.unpack('<?', f.read(1))[0]
    elif vtype == 8: return read_string()
    elif vtype == 9:
        atype = struct.unpack('<I', f.read(4))[0]; alen = struct.unpack('<Q', f.read(8))[0]
        return [read_value(atype) for _ in range(alen)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    else: raise ValueError(f'Unknown type {vtype}')

for i in range(n_kv): key = read_string(); vtype = struct.unpack('<I', f.read(4))[0]; val = read_value(vtype)

tensor_map = {}
for i in range(n_tensors):
    name = read_string(); ndims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
    ttype = struct.unpack('<I', f.read(4))[0]; offset = struct.unpack('<Q', f.read(8))[0]
    tensor_map[name] = (dims, ttype, offset)

pos_now = f.tell()
data_start = (pos_now + 31) & ~31

def dequant_q4_0_row(offset, n_elements):
    nb = n_elements // 32
    f.seek(data_start + offset)
    out = np.zeros(n_elements, dtype=np.float32)
    for b in range(nb):
        d_raw = struct.unpack('<H', f.read(2))[0]
        d = float(np.frombuffer(struct.pack('<H', d_raw), dtype=np.float16)[0])
        qs = f.read(16)
        for j in range(16):
            byte = qs[j]
            lo = (byte & 0x0F) - 8
            hi = (byte >> 4) - 8
            out[b * 32 + 2 * j] = lo * d
            out[b * 32 + 2 * j + 1] = hi * d
    return out

def get_f32_tensor(offset, n_elements):
    f.seek(data_start + offset)
    return np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()

# 1. Embed token 32010
row_bytes_q4_0 = (dim // 32) * 18
embd_offset = tensor_map['token_embd.weight'][2]
emb = dequant_q4_0_row(embd_offset + token_id * row_bytes_q4_0, dim)
print(f"Embed: min={emb.min():.4f} max={emb.max():.4f} L2_sq={np.sum(emb*emb):.4f}")

# 2. RMSNorm
norm_offset = tensor_map['blk.0.attn_norm.weight'][2]
attn_norm_w = get_f32_tensor(norm_offset, dim)
ss = np.mean(emb * emb)
scale = 1.0 / np.sqrt(ss + 1e-6)
xn = emb * scale * attn_norm_w
print(f"xn after RMSNorm: L2_sq={np.sum(xn*xn):.4f} absmax={np.max(np.abs(xn)):.4f}")
print(f"  xn[0:8] = {xn[:8]}")

# 3. QKV projection
qkv_offset = tensor_map['blk.0.attn_qkv.weight'][2]
# Q is first 3072 rows, K is next 3072, V is last 3072
# Each row is 3072 Q4_0 elements

print("\nComputing Q[0..7] (first 8 elements of Q projection)...")
q_vals = np.zeros(8, dtype=np.float32)
for i in range(8):
    row = dequant_q4_0_row(qkv_offset + i * row_bytes_q4_0, dim)
    q_vals[i] = np.dot(row, xn)
    
print(f"Python Q[0..7] = {q_vals}")

# Compare with TensorOS: Q[0..3] = -0.0141, 0.0024, -0.0243, -0.0079  
print(f"\nTensorOS Q[0..3] = -0.0141, 0.0024, -0.0243, -0.0079")
print(f"Python   Q[0..3] = {q_vals[0]:.4f}, {q_vals[1]:.4f}, {q_vals[2]:.4f}, {q_vals[3]:.4f}")

# Now compute Q_L2 (sum of squares of all 3072 Q elements)
print("\nComputing full Q vector (3072 elements)...")
q_full = np.zeros(dim, dtype=np.float32)
for i in range(dim):
    row = dequant_q4_0_row(qkv_offset + i * row_bytes_q4_0, dim)
    q_full[i] = np.dot(row, xn)
    if i % 500 == 0:
        print(f"  ... row {i}/{dim}")

q_l2_sq = np.sum(q_full * q_full)
print(f"\nPython Q_L2_sq = {q_l2_sq:.4f}")
print(f"TensorOS Q_L2 = 2.4324")

# Also do K (next 3072 rows)
print("\nComputing K[0..3]...")
k_base_offset = qkv_offset + dim * row_bytes_q4_0
k_vals = np.zeros(4, dtype=np.float32)
for i in range(4):
    row = dequant_q4_0_row(k_base_offset + i * row_bytes_q4_0, dim)
    k_vals[i] = np.dot(row, xn)
print(f"Python K[0..3] = {k_vals}")
print(f"TensorOS K[0..3] = 0.0314, -0.0079, 0.0257, 0.0118")

f.close()
print("\nDone!")
