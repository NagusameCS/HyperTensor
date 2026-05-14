# hypertensor-runtime

Installable Python package wrapping the **geodessical** C binary and the
**libhypercore** shared library so they can be installed via pip alongside the
rest of the HyperTensor ecosystem:

| Package | Role |
|---|---|
| `hypertensor-framework` | Pure-Python research framework |
| `hypertensor-core`      | Python bindings + core algorithms |
| `hypertensor-runtime`   | **This package** — native binaries |
| `ht-repro`              | Reproduction CLI + web UI |

## Install

```bash
pip install hypertensor-runtime
geodessical --help
```

If a binary is not bundled for your platform you'll get a clear error pointing
at the build instructions.

## Building wheels per platform

Bundled binaries live under `hypertensor_runtime/bin/<platform>/` where
`<platform>` is e.g. `linux-x86_64`, `win-amd64`, `mac-arm64`.

Local build (Windows host):

```powershell
cd ..
.\build_host.ps1
mkdir -p hypertensor_runtime\bin\win-amd64
copy build_host\geodessical.exe hypertensor_runtime\bin\win-amd64\
copy build_host\hypercore.dll   hypertensor_runtime\bin\win-amd64\
python -m build --wheel hypertensor_runtime
```

Linux / macOS (via cibuildwheel):

```bash
cibuildwheel --config-file deploy/cibuildwheel.toml hypertensor_runtime
```

## Programmatic access

```python
from hypertensor_runtime import geodessical_path, libhypercore_path
print(geodessical_path())     # absolute Path to the bundled binary
print(libhypercore_path())    # absolute Path to the shared lib
```

`ht_repro.runtime_loader` will pick these up automatically when both packages
are installed.
