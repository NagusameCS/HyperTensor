#!/usr/bin/env python3
"""Full L0 verification using gguf library for proper tensor loading."""
import struct, numpy as np, sys
from gguf import GGUFReader

GGUF = r"C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf"
PROMPT_TOKENS = [32010, 13, 5618, 338, 385, 13598, 1788, 29973, 32007, 13, 32001, 13]

dim = 3072
n_heads = 32
n_kv = 32
hd = 96
ff_dim = 8192

def fp16_to_f32(h):
    return np.float32(np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0])

def dequant_q4_0_block(data, offset):
    """Dequant one Q4_0 block (32 elements from 18 bytes)"""
    d = fp16_to_f32(struct.unpack_from('<H', data, offset)[0])
    out = np.zeros(32, dtype=np.float32)
    for j in range(16):
        byte = data[offset + 2 + j]
        lo = (byte & 0x0F) - 8
        hi = ((byte >> 4) & 0x0F) - 8
        out[j] = lo * d
        out[j + 16] = hi * d
    return out

def dequant_q4_1_block(data, offset):
    """Dequant one Q4_1 block (32 elements from 20 bytes)"""
    d = fp16_to_f32(struct.unpack_from('<H', data, offset)[0])
    m = fp16_to_f32(struct.unpack_from('<H', data, offset + 2)[0])
    out = np.zeros(32, dtype=np.float32)
    for j in range(16):
        byte = data[offset + 4 + j]
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        out[j] = lo * d + m
        out[j + 16] = hi * d + m
    return out

def dequant_q4_0(raw_bytes, n_elements):
    nb = n_elements // 32
    out = np.zeros(n_elements, dtype=np.float32)
    for b in range(nb):
        out[b*32:(b+1)*32] = dequant_q4_0_block(raw_bytes, b * 18)
    return out

def dequant_q4_1(raw_bytes, n_elements):
    nb = n_elements // 32
    out = np.zeros(n_elements, dtype=np.float32)
    for b in range(nb):
        out[b*32:(b+1)*32] = dequant_q4_1_block(raw_bytes, b * 20)
    return out

def gemv_q4_0(weight_data, x, out_dim, in_dim):
    """Matrix-vector multiply for Q4_0 weight: out = W @ x"""
    # Weight is row-major: out_dim rows of in_dim elements
    out = np.zeros(out_dim, dtype=np.float32)
    nb_row = in_dim // 32
    for r in range(out_dim):
        row_offset = r * nb_row * 18
        acc = 0.0
        for b in range(nb_row):
            block = dequant_q4_0_block(weight_data, row_offset + b * 18)
            acc += np.dot(block, x[b*32:(b+1)*32])
        out[r] = acc
    return out

def gemv_q4_1(weight_data, x, out_dim, in_dim):
    """Matrix-vector multiply for Q4_1 weight: out = W @ x"""
    out = np.zeros(out_dim, dtype=np.float32)
    nb_row = in_dim // 32
    for r in range(out_dim):
        row_offset = r * nb_row * 20
        acc = 0.0
        for b in range(nb_row):
            block = dequant_q4_1_block(weight_data, row_offset + b * 20)
            acc += np.dot(block, x[b*32:(b+1)*32])
        out[r] = acc
    return out

def rmsnorm(x, w, eps=1e-5):
    ss = np.mean(x * x) + eps
    return x * (1.0 / np.sqrt(ss)) * w

def rope(vec, pos, rope_base=10000.0, factors=None):
    hd = len(vec)
    out = vec.copy()
    for i in range(0, hd, 2):
        freq = 1.0 / (rope_base ** (float(i) / float(hd)))
        if factors is not None:
            freq /= factors[i // 2]
        theta = pos * freq
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        v0, v1 = out[i], out[i+1]
        out[i]   = v0 * cos_t - v1 * sin_t
        out[i+1] = v0 * sin_t + v1 * cos_t
    return out

def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)

print("Loading GGUF...")
reader = GGUFReader(GGUF)

# Build tensor lookup
tensors = {}
for t in reader.tensors:
    tensors[t.name] = t

# Get rope factors
rope_short = tensors["rope_factors_short.weight"].data.copy()
print(f"Rope short[0:5]: {rope_short[:5]}")

# We need raw bytes for quantized dot products
# For gguf library, tensor.data gives dequantized values for F32 type
# For quantized types, we need the raw bytes

# Let's just dequantize the full matrices we need
print("Loading weight matrices (this may take a while for Q4_0)...")

# token_embd: Q4_0, [3072, 32064] in GGUF (rows=32064 vocab, cols=3072 dim)
# Actually in GGUF, dims are stored as [ne0, ne1] = [3072, 32064]
# which means: 32064 rows of 3072 columns = [32064, 3072] in row-major
embd_t = tensors["token_embd.weight"]
print(f"  token_embd: shape={embd_t.shape}, type={embd_t.tensor_type}")

# For Q4_0, the gguf lib data field might just be the raw quants
# Let's check if we can get dequantized data
embd_data = embd_t.data  # This might be raw quant data

# For F32 tensors, .data gives float32 array directly
attn_norm = tensors["blk.0.attn_norm.weight"].data.copy()
ffn_norm = tensors["blk.0.ffn_norm.weight"].data.copy()
print(f"  attn_norm: shape={attn_norm.shape}")

# For the actual GEMV, let me just dequantize on the fly
# Actually let's use a simpler approach: just test embedding + rmsnorm + first few Q values
# by doing the dot product row by row

# Get embedding for token 32010
# embd raw data is Q4_0 with [3072, 32064] = 32064 rows of 3072
# Each row = 3072/32 = 96 blocks = 96*18 = 1728 bytes
tok = PROMPT_TOKENS[0]  # 32010
row_bytes = (dim // 32) * 18  # 1728

# Read raw data
import mmap, os
with open(GGUF, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    # Find tensor data offset - need the data section start
    # The gguf reader should have this info
    # Actually, let me compute it from the tensor offset
    # tensor.data_offset gives the offset from data section start
    
    embd_offset = embd_t.data_offset
    # Find data section start by looking at the file structure
    # data_start = file size - total tensor data size? 
    # Actually easier: first tensor's offset from file start
    
    # The GGUFReader stores tensors with their field data
    # Let me just get the data from the reader
    
    # For Q4_0 type=2, the .data field of gguf tensors returns the raw quant blocks
    # Let me check what we actually get
    d = embd_t.data
    print(f"  embd_t.data type={type(d)}, dtype={d.dtype if hasattr(d,'dtype') else 'N/A'}, shape={d.shape if hasattr(d,'shape') else len(d)}")
    
    mm.close()

# The gguf library returns raw quant data for quantized types
# Need to dequantize manually
def dequant_tensor(t):
    """Dequantize a gguf tensor to float32"""
    ttype = t.tensor_type.value if hasattr(t.tensor_type, 'value') else int(t.tensor_type)
    n_elem = t.n_elements
    raw = bytes(t.data)
    
    if ttype == 0:  # F32
        return np.frombuffer(raw, dtype=np.float32).copy()
    elif ttype == 2:  # Q4_0
        return dequant_q4_0(raw, n_elem)
    elif ttype == 3:  # Q4_1
        return dequant_q4_1(raw, n_elem)
    else:
        raise ValueError(f"Unsupported type {ttype}")

# Embedding
embd_flat = dequant_tensor(embd_t)
embd_mat = embd_flat.reshape(32064, dim)
x = embd_mat[tok].copy()
print(f"\nEmbedding tok={tok}: min={x.min():.4f} max={x.max():.4f} abssum={np.abs(x).sum():.2f}")

# RMSNorm
xn = rmsnorm(x, attn_norm)
print(f"After RMSNorm: min={xn.min():.6f} max={xn.max():.6f}")

# QKV projection
qkv_t = tensors["blk.0.attn_qkv.weight"]
qkv_flat = dequant_tensor(qkv_t)
qkv_mat = qkv_flat.reshape(dim * 3, dim)  # [9216, 3072]
qkv = qkv_mat @ xn
print(f"QKV: min={qkv.min():.6f} max={qkv.max():.6f}")

q = qkv[:dim].copy()
k = qkv[dim:2*dim].copy()
v = qkv[2*dim:3*dim].copy()
print(f"Q[0:4]: {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
print(f"Q L2: {np.sum(q*q):.4f}")

# Apply RoPE at pos=0 (should be identity)
for h in range(n_heads):
    q[h*hd:(h+1)*hd] = rope(q[h*hd:(h+1)*hd], 0, 10000.0, rope_short)
for h in range(n_kv):
    k[h*hd:(h+1)*hd] = rope(k[h*hd:(h+1)*hd], 0, 10000.0, rope_short)

print(f"Q after RoPE pos=0 [0:4]: {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")

# Attention: just one head (self-attention at pos=0, seq_len=1)
scale = 1.0 / np.sqrt(float(hd))
# score = Q[h=0] . K[h=0] / sqrt(96)
# at pos=0, there's only 1 past token (itself), so score = softmax([X]) = [1.0]
# attn_out = 1.0 * V[h=0]
attn_out = v.copy()  # At pos=0, each head just gets its own V

# Output projection
attn_o_t = tensors["blk.0.attn_output.weight"]
attn_o_flat = dequant_tensor(attn_o_t)
attn_o_mat = attn_o_flat.reshape(dim, dim)
o = attn_o_mat @ attn_out
print(f"Output proj: min={o.min():.6f} max={o.max():.6f}")

# Residual
x = x + o
print(f"After attn residual: min={x.min():.4f} max={x.max():.4f} abssum={np.abs(x).sum():.2f}")

# FFN
xn2 = rmsnorm(x, ffn_norm)

# FFN up (fused gate+up): [16384, 3072]
ffn_up_t = tensors["blk.0.ffn_up.weight"]
ffn_up_flat = dequant_tensor(ffn_up_t)
ffn_up_mat = ffn_up_flat.reshape(ff_dim * 2, dim)  # [16384, 3072]
ffn_raw = ffn_up_mat @ xn2
gate = ffn_raw[:ff_dim]
up = ffn_raw[ff_dim:]

# SwiGLU
ffn_out = silu(gate) * up

# FFN down
ffn_down_t = tensors["blk.0.ffn_down.weight"]
print(f"  ffn_down: shape={ffn_down_t.shape}, type={ffn_down_t.tensor_type}")
ffn_down_flat = dequant_tensor(ffn_down_t)
ffn_down_mat = ffn_down_flat.reshape(dim, ff_dim)  # [3072, 8192]
down = ffn_down_mat @ ffn_out

# Residual
x = x + down

print(f"\n=== After L0 at pos=0 (token={PROMPT_TOKENS[0]}) ===")
print(f"  min={x.min():.4f} max={x.max():.4f} abssum={np.abs(x).sum():.2f}")
print(f"  x[0:4]={x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {x[3]:.6f}")
print(f"  x[3068:3072]={x[3068]:.6f} {x[3069]:.6f} {x[3070]:.6f} {x[3071]:.6f}")

# Also check TensorOS debug: "after L0: min=-0.3718 max=0.3708 abssum=125.45"
print(f"\nTensorOS reports: after L0: min=-0.3718 max=0.3708 abssum=125.45")
print(f"Python computed:  after L0: min={x.min():.4f} max={x.max():.4f} abssum={np.abs(x).sum():.2f}")
