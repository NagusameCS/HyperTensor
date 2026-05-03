# legacy/

This directory holds the freestanding-OS code that the project started life as
(originally TensorOS). It is **not** part of the HyperTensor inference runtime
and is **not** built by [build_host.ps1](../build_host.ps1).

It is kept here because:

- it is the historical origin of several ideas now in `runtime/nn/` (axiom
  discovery, manifold extraction);
- some of it is referenced from research notes;
- removing it would lose git history pointers for those origins.

## What's in here

| Path           | Original purpose |
| ---            | --- |
| `boot/`        | x86_64 / arm64 boot stubs and linker scripts for the freestanding kernel. |
| `kernel/`      | Freestanding-OS kernel: scheduler, mm, drivers, fs, ipc, security, network. |
| `virt/`        | QEMU virtualisation glue. |
| `userland/`    | Early userspace tooling (deploy, monitor, shell, train). |
| `axiom_vis/`   | Per-model axiom-discovery visualisation dumps (smollm2-135m, phi-3.5-mini, gemma-4-e2b). |
| `Makefile`     | The kernel build (boot+kernel into ISO). Not used for the host runtime. |

## What HyperTensor actually builds

The C11 inference runtime lives in [host/](../host/) and [runtime/](../runtime/),
built by [build_host.ps1](../build_host.ps1) into `build_host/geodessical.exe`.
Nothing under `legacy/` is compiled, linked, or required at runtime.

If you are looking for current code, look one directory up.
