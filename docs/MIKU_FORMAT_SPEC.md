# .MIKU File Format Specification v1

## Overview

The `.miku` format --- named after Hatsune Miku, the most iconic vocaloid ---
bundles a language model with its HyperTensor manifold state, enabling a
"living model" that grows through conversation. It is the first file format
designed for models that learn from interaction without backpropagation ---
the manifold expands organically through Jacobi integration of novel
conversational trajectories.

Just as Miku's voice synthesizes entirely new songs from a fixed vocal
library, a `.miku` file synthesizes new knowledge from a fixed-weight model
through geometric manifold expansion.

Papers: XI (UGT), XIII (Safe OGD), XIV (Snipe), XV (COG+TEH)

## Rationale

Standard model formats (`.safetensors`, `.gguf`, `.pt`) capture static weights.
They say nothing about:
- The taxonomic basis that organizes the model's internal coordinate system
- The forbidden behavioral coordinates that are geometrically blocked
- The living manifold metric that grows with every conversation
- The trajectory cache that enables cross-session memory

A `.miku` file is a living checkpoint. Load it with HyperChat and the model
continues growing from where it left off.

## Format Structure

```
model.miku          --- JSON metadata (human-readable)
model.miku.pt       --- Tensor binary blob (PyTorch)
```

### JSON Schema

```json
{
  "format": "miku-v1",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "k_ugt": 512,
  "d_model": 3584,
  "created": "2026-05-03T15:30:00Z",
  "papers": "XI-XV",
  "forbidden_coords": [120, 28, 476, 196, 466, ...],
  "snipe_coords": [230, 25, 256, 71, 169, 87],
  "trajectories": [
    {
      "proj": [0.12, -0.34, ...],
      "label": "quantum computing superposition",
      "time": 1714752000.0
    }
  ],
  "conversation_log": [
    {
      "user": "Tell me about the Riemann Hypothesis",
      "response": "The Riemann Hypothesis is...",
      "teh": 0.0,
      "cog": "EXPANDED",
      "sim": 0.89,
      "metric": 0.031,
      "traj": 14,
      "ms": 7227
    }
  ]
}
```

## Tensor Blob

```python
{
    "basis": torch.Tensor,     # [d_model, k_ugt] --- UGT taxonomic basis
    "metric": torch.Tensor,    # [k_ugt, k_ugt] --- COG living manifold metric
}
```

## Key Design Decisions

1. JSON metadata + PyTorch tensors --- Metadata is human-readable and diffable.
   Tensors are in PyTorch's native format for fast GPU upload.

2. Basis is the core innovation --- Unlike standard formats that only capture
   weights, the basis defines the model's coordinate system. Two models with
   different bases will have different behavioral coordinate maps.

3. Trajectories are privacy-sensitive --- The trajectory cache contains
   embedding vectors of user conversations. Applications should consider
   encryption or local-only storage.

4. Metric is the "living" part --- The COG metric tensor changes with every
   novel interaction. Loading a `.miku` file restores the learned manifold
   geometry.

## Why Not Extend safetensors/GGUF?

- safetensors: Captures weights only. No metadata extensibility for manifold state.
- GGUF: Quantized weights + tokenizer config. No support for k-manifold structures.
- `.miku` fills a genuine gap: No existing format supports a model that
  changes its geometric structure through use. The metric tensor is not a
  weight --- it's a learned Riemannian metric on the k-manifold.

## Why ".MIKU"?

Named after Hatsune Miku, the virtual singer who proved that a fixed synthesis
engine can generate infinite novel creative works. The analogy is precise:
- Miku's voicebank = the model's frozen weights
- Each new song = each COG manifold expansion
- The vocaloid software = the HyperTensor stack (UGT + Safe OGD + Snipe + COG)
- A `.miku` file = a saved session with all its creative growth

## Future Extensions (miku-v2)

- Compressed trajectory storage (quantization + dedup)
- Multi-user trajectory isolation
- Basis hot-swap (load one model's basis onto another for UGT compatibility)
- Encrypted conversation log for privacy
- Tokenizer snapshot bundled alongside
- Native model weights (when Native Geodesic Training matures)

## Status

Draft --- implemented in `scripts/hyper_chat.py` as of 2026-05-03.
Used by the HyperChat CLI for interactive sessions on the HyperTensor living model.
