/* =============================================================================
 * TensorOS - AI Shell (aishell)
 * =============================================================================
 * The default interactive shell for TensorOS. Unlike traditional shells that
 * execute programs, aishell is AI-first:
 *
 *   - Commands can be natural language (JIT-compiled via Pseudocode runtime)
 *   - Native tensor inspection / model management commands
 *   - Built-in git (kernel-level)
 *   - Pipeline syntax for model chaining: model1 |> model2 |> deploy
 *   - Tab-completion aware of loaded models, datasets, and tensor shapes
 *
 * Built-in commands:
 *   model load <name>        Load a model into an MEU
 *   model list               List running MEUs
 *   model info <id>          Show MEU stats (FLOPS, memory, latency)
 *   model kill <id>          Terminate an MEU
 *   tensor shape <expr>      Print tensor shape
 *   tensor cast <id> <dtype> Requantize a tensor
 *   infer <model> <input>    Run inference
 *   train <model> <dataset>  Launch training MEU
 *   deploy <model> [port]    Deploy model as service
 *   git <subcommand>         Kernel-level git
 *   pkg install <model>      Install from model registry
 *   pkg search <query>       Search registries
 *   monitor                  Open tensor monitor
 *   sandbox <policy> <cmd>   Run command in sandbox
 *   help                     Show help
 *   exit                     Shutdown
 * =============================================================================*/

#ifndef AISHELL_H
#define AISHELL_H

#include "kernel/core/kernel.h"
#include "runtime/pseudocode/pseudocode_jit.h"
#include "kernel/fs/git.h"
#include "pkg/modelpkg.h"
#include "kernel/security/sandbox.h"

#define SHELL_MAX_LINE    256
#define SHELL_MAX_ARGS    32
#define SHELL_MAX_HISTORY 16
#define SHELL_PROMPT_MAX  64

typedef struct {
    char lines[SHELL_MAX_HISTORY][SHELL_MAX_LINE];
    int  count;
    int  cursor;
} shell_history_t;

typedef struct {
    char          prompt[SHELL_PROMPT_MAX];
    shell_history_t history;
    pseudo_runtime_t *runtime;   /* Pseudocode JIT for scripting */
    bool          running;
    bool          interactive;
    uint32_t      commands_executed;
    uint64_t      session_start_ticks;
} aishell_t;

#endif /* AISHELL_H */
