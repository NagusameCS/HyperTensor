# TensorChat Native (Windows C App)

TensorChat Native is a modular desktop shell written in C for Windows.

Current features:
- warmup loading screen on startup
- OBS-style module controls (Add Chat, Add Editor, Add Terminal, Add Outline)
- Chat as the default module (history, input, send)
- optional editor and structure-outline panels
- optional sandbox terminal panel
- Jekyll-inspired theme presets (Minima, Cayman, Architect, Slate, Midnight, Hacker, Dinky)

## Build

From this folder:

```
./build_native.ps1
```

Output: `tensorchat_native.exe`

## Notes

- This is a hosted Windows tool and is not part of the kernel build.
- Chat calls the TensorOS HTTP API at `http://127.0.0.1:8080/v1/chat/completions`, with local fallback when the runtime is unavailable.
- Outline parsing tracks common symbols (`int`, `void`, `class`, `struct`, `fn`, `def`).
- The terminal module runs in sandbox mode with built-in commands (`help`, `clear`, `pwd`, `status`).
- This module is the UI shell for future runtime integration and richer theming.
