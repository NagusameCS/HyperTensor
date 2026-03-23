/* pseudo_stdlib.h - Pseudocode Standard Library */
#ifndef PSEUDO_STDLIB_H
#define PSEUDO_STDLIB_H

#include "runtime/pseudocode/pseudocode_jit.h"

/* Built-in function signature: takes runtime + args → returns value */
typedef runtime_value_t (*pseudo_builtin_fn)(pseudo_runtime_t *rt,
                                              runtime_value_t *args,
                                              int n_args);

typedef struct {
    const char       *name;
    pseudo_builtin_fn fn;
    int               min_args;
    int               max_args;
} pseudo_builtin_t;

/* Register all stdlib builtins into the runtime */
int  pseudo_stdlib_init(pseudo_runtime_t *rt);

/* Look up a builtin by name; returns NULL if not found */
const pseudo_builtin_t *pseudo_stdlib_lookup(const char *name);

/* Call a builtin by name with arguments */
runtime_value_t pseudo_stdlib_call(pseudo_runtime_t *rt,
                                    const char *name,
                                    runtime_value_t *args,
                                    int n_args);

#endif /* PSEUDO_STDLIB_H */
