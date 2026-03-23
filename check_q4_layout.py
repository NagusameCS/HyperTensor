"""Compare standard vs interleaved Q4_0 dequantization."""
import struct, numpy as np
from gguf import GGUFReader

GGUF = r"C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf"
reader = GGUFReader(GGUF)
for t in reader.tensors:
    if t.name == "token_embd.weight":
        raw = bytes(t.data[0,:18])  # First block of first row (18 bytes)
        break

d_raw = struct.unpack_from('<H', raw, 0)[0]
d = float(np.frombuffer(struct.pack('<H', d_raw), dtype=np.float16)[0])
print(f"d = {d}")

# Method A: standard (lo in first 16, hi in last 16)
out_a = [0.0] * 32
for j in range(16):
    byte = raw[2 + j]
    lo = (byte & 0x0F) - 8
    hi = ((byte >> 4) & 0x0F) - 8
    out_a[j] = lo * d
    out_a[j + 16] = hi * d

# Method B: interleaved (TensorOS uses this)
out_b = [0.0] * 32
for j in range(16):
    byte = raw[2 + j]
    lo = (byte & 0x0F) - 8
    hi = ((byte >> 4) & 0x0F) - 8
    out_b[2*j] = lo * d
    out_b[2*j + 1] = hi * d

print(f"Method A (standard): {[round(x,4) for x in out_a[:8]]}...{[round(x,4) for x in out_a[16:20]]}")
print(f"Method B (interleaved): {[round(x,4) for x in out_b[:8]]}...{[round(x,4) for x in out_b[16:20]]}")
print(f"A L2: {sum(x*x for x in out_a):.6f}")
print(f"B L2: {sum(x*x for x in out_b):.6f}")
print(f"Same L2: {abs(sum(x*x for x in out_a) - sum(x*x for x in out_b)) < 1e-6}")
print()

# Now let's check: which method matches what llama.cpp/GGML uses?
# The standard GGML dequant_q4_0 from ggml-quants.c:
# x[i + 0]  = (qs[i/2] & 0x0F) - 8   (low nibble, even index)
# x[i + qk/2] = (qs[i/2] >> 4) - 8     (high nibble, at offset 16)
# Wait, actually GGML uses:
#   for j = 0..qk/2-1:
#     x[j]         = ((qs[j] & 0x0F) - 8) * d
#     x[j + qk/2]  = ((qs[j] >>  4) - 8) * d
# where qk=32, so:
#   x[j] for j=0..15: low nibble of qs[j]
#   x[j+16] for j=0..15: high nibble of qs[j]
# This is METHOD A (standard).
print("GGML reference: Method A (lo in [0:16], hi in [16:32])")
print("TensorOS uses: Method B (interleaved)")
print()

# Check which method gives the correct embedding for token 32010
# TensorOS reports: embed tok=32010: min=-0.1216 max=0.2158 abssum=72.35
# Let's compute with both methods

for t2 in reader.tensors:
    if t2.name == "token_embd.weight":
        embd_data = t2.data
        break

dim = 3072
tok = 32010
nb = dim // 32

# Get raw bytes for this token's row
row_raw = bytes(embd_data[tok])

embed_a = np.zeros(dim, dtype=np.float32)
embed_b = np.zeros(dim, dtype=np.float32)

for b in range(nb):
    off = b * 18
    d_val = float(np.frombuffer(row_raw[off:off+2], dtype=np.float16)[0])
    for j in range(16):
        byte = row_raw[off + 2 + j]
        lo = (byte & 0x0F) - 8
        hi = ((byte >> 4) & 0x0F) - 8
        # Method A
        embed_a[b*32 + j] = lo * d_val
        embed_a[b*32 + j + 16] = hi * d_val
        # Method B
        embed_b[b*32 + 2*j] = lo * d_val
        embed_b[b*32 + 2*j + 1] = hi * d_val

print(f"Method A embed: min={embed_a.min():.4f} max={embed_a.max():.4f} abssum={np.abs(embed_a).sum():.2f}")
print(f"Method B embed: min={embed_b.min():.4f} max={embed_b.max():.4f} abssum={np.abs(embed_b).sum():.2f}")
print(f"TensorOS embed: min=-0.1216 max=0.2158 abssum=72.35")
print()
print(f"Method A embed[0:4]: {embed_a[0]:.6f} {embed_a[1]:.6f} {embed_a[2]:.6f} {embed_a[3]:.6f}")
print(f"Method B embed[0:4]: {embed_b[0]:.6f} {embed_b[1]:.6f} {embed_b[2]:.6f} {embed_b[3]:.6f}")
