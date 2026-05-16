# civilized-HyperTensor Structure

## Overview

This repository keeps the runnable implementation surfaces from HyperTensor and strips out archival publication, shell-wrapper clutter, and non-self-contained build dependencies.

## What Remains

### Core Engineering

- `runtime/` - runtime sources kept for reference and future expansion; not part of the default build path
- `hypercore/` - core mathematical implementation layer
- `lib/` - shared utilities
- `tests/` - native test suite

### Implementation Surfaces

- `examples/` - runnable demonstrations organized by original, domain, and technique surfaces
- `scripts/` - algorithmic implementations and verification utilities
- `configs/` - model and math configuration inputs
- `data/` - benchmark and verification datasets
- `benchmarks/` - benchmark configuration and registry data
- `repro/` - reproduction guides and expected-output helpers

### Build & Packaging

- `Dockerfile` - containerized reproducible environment
- `CMakeLists.txt` - native build configuration
- `pyproject.toml` - project metadata
- `setup.py` - editable/local packaging support

## What Was Removed

- `docs/` - generated and narrative documentation trees
- `ARXIV_SUBMISSIONS/` - paper submission archives and publication bundles
- shell orchestration wrappers for remote runs, paper builds, and legacy training
- runtime build dependencies that are not present in this trimmed checkout

## Layout Rationale

1. Keep the repo centered on code that executes, tests, or supports reproducible implementation work.
2. Remove publication-heavy and environment-specific material that does not help day-to-day development.
3. Preserve the core implementation and verification surfaces so the repository remains a working research codebase, not an archive.

## Verification Pointers

- Build the native core with CMake and the Dockerfile.
- Run the Python verification scripts in `scripts/`.
- Run the examples and native tests that remain in the repo.
