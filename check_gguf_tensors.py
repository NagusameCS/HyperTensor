import struct

f = open(r'C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf', 'rb')
magic = f.read(4)
version = struct.unpack('<I', f.read(4))[0]
n_tensors = struct.unpack('<Q', f.read(8))[0]
n_kv = struct.unpack('<Q', f.read(8))[0]

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
        atype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        return [read_value(atype) for _ in range(alen)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    else: raise ValueError(f'Unknown type {vtype}')

for i in range(n_kv):
    key = read_string()
    vtype = struct.unpack('<I', f.read(4))[0]
    val = read_value(vtype)

# Read ALL tensor names
bias_tensors = []
all_tensors = []
for i in range(n_tensors):
    name = read_string()
    ndims = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
    ttype = struct.unpack('<I', f.read(4))[0]
    offset = struct.unpack('<Q', f.read(8))[0]
    all_tensors.append((name, dims, ttype, offset))
    if 'bias' in name.lower():
        bias_tensors.append((name, dims, ttype))

if bias_tensors:
    print(f'Found {len(bias_tensors)} bias tensors:')
    for name, dims, ttype in bias_tensors:
        print(f'  {name}: dims={dims} type={ttype}')
else:
    print('NO bias tensors found in GGUF file!')

# blk.0 tensors
blk0 = [(n, d, t) for n, d, t, o in all_tensors if n.startswith('blk.0.')]
print(f'\nblk.0 tensor names ({len(blk0)}):')
for n, d, t in blk0:
    print(f'  {n}: dims={d} type={t}')

# Global tensors
glb = [(n, d, t) for n, d, t, o in all_tensors if not n.startswith('blk.')]
print(f'\nGlobal tensors ({len(glb)}):')
for n, d, t in glb:
    print(f'  {n}: dims={d} type={t}')

print(f'\nTotal: {n_tensors} tensors')
f.close()
