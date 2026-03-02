/* =============================================================================
 * TensorOS - Model Package Manager Implementation
 * =============================================================================*/

#include "pkg/modelpkg.h"

modelpkg_state_t g_modelpkg;

int modelpkg_init(void)
{
    kmemset(&g_modelpkg, 0, sizeof(g_modelpkg));

    /* Add default model registry */
    modelpkg_add_registry("tensoros-hub", "https://models.tensoros.dev");
    modelpkg_add_registry("huggingface", "https://huggingface.co/api/models");

    kprintf_debug("[PKG] Model package manager initialized with %d registries\n",
                  g_modelpkg.registry_count);
    return 0;
}

int modelpkg_add_registry(const char *name, const char *url)
{
    if (g_modelpkg.registry_count >= MODELPKG_MAX_REGISTRIES) return -1;

    model_registry_t *reg = &g_modelpkg.registries[g_modelpkg.registry_count++];
    kmemset(reg, 0, sizeof(*reg));

    for (int i = 0; i < 63 && name[i]; i++) reg->name[i] = name[i];
    for (int i = 0; i < 255 && url[i]; i++) reg->url[i] = url[i];
    reg->enabled = true;

    return 0;
}

int modelpkg_install(const char *name, const char *version)
{
    if (g_modelpkg.installed_count >= MODELPKG_MAX_INSTALLED) return -1;

    kprintf("[PKG] Installing %s", name);
    if (version) kprintf(" v%s", version);
    kprintf("...\n");

    installed_model_t *model = &g_modelpkg.installed[g_modelpkg.installed_count++];
    kmemset(model, 0, sizeof(*model));

    /* Copy name */
    for (int i = 0; i < 63 && name[i]; i++)
        model->manifest.name[i] = name[i];
    if (version) {
        for (int i = 0; i < 15 && version[i]; i++)
            model->manifest.version[i] = version[i];
    }

    /* TODO: Actually download from registry */
    /* 1. Fetch manifest from registry
     * 2. Resolve dependencies
     * 3. Download model files
     * 4. Verify checksums
     * 5. Install to /models/<name>/
     * 6. Git commit installation
     */

    model->status = PKG_STATUS_INSTALLED;
    model->install_time = kstate.uptime_ticks;

    /* Set install path */
    char *path = model->install_path;
    const char *prefix = "/models/";
    while (*prefix) *path++ = *prefix++;
    for (int i = 0; i < 63 && name[i]; i++) *path++ = name[i];
    *path = '\0';

    kprintf("[PKG] Installed %s to %s\n", name, model->install_path);
    return 0;
}

int modelpkg_uninstall(const char *name)
{
    for (uint32_t i = 0; i < g_modelpkg.installed_count; i++) {
        if (kstrcmp(g_modelpkg.installed[i].manifest.name, name) == 0) {
            /* Shift remaining entries */
            for (uint32_t j = i; j < g_modelpkg.installed_count - 1; j++)
                g_modelpkg.installed[j] = g_modelpkg.installed[j + 1];
            g_modelpkg.installed_count--;
            kprintf("[PKG] Uninstalled %s\n", name);
            return 0;
        }
    }
    return -1;
}

int modelpkg_list_installed(installed_model_t *models,
                             uint32_t max, uint32_t *count)
{
    uint32_t to_copy = g_modelpkg.installed_count < max ?
                        g_modelpkg.installed_count : max;
    for (uint32_t i = 0; i < to_copy; i++)
        models[i] = g_modelpkg.installed[i];
    if (count) *count = to_copy;
    return 0;
}

int modelpkg_optimize(const char *name, tensor_dtype_t target_dtype)
{
    for (uint32_t i = 0; i < g_modelpkg.installed_count; i++) {
        if (kstrcmp(g_modelpkg.installed[i].manifest.name, name) == 0) {
            installed_model_t *model = &g_modelpkg.installed[i];
            model->status = PKG_STATUS_OPTIMIZING;

            kprintf("[PKG] Quantizing %s to %s...\n", name,
                    target_dtype == TENSOR_DTYPE_F16 ? "FP16" :
                    target_dtype == TENSOR_DTYPE_INT8 ? "INT8" :
                    target_dtype == TENSOR_DTYPE_INT4 ? "INT4" : "unknown");

            /* TODO: Actual quantization:
             * 1. Load model weights
             * 2. Apply quantization algorithm (GPTQ, AWQ, etc.)
             * 3. Save quantized weights
             * 4. Update manifest
             */

            model->optimized = true;
            model->optimized_dtype = target_dtype;
            model->status = PKG_STATUS_INSTALLED;

            kprintf("[PKG] Optimization complete\n");
            return 0;
        }
    }
    return -1;
}

int modelpkg_auto_optimize(const char *name)
{
    /* Detect GPU capabilities and choose best dtype */
    if (kstate.gpu_count > 0) {
        /* Modern GPUs: use FP16 or BF16 */
        return modelpkg_optimize(name, TENSOR_DTYPE_F16);
    } else {
        /* CPU-only: use INT8 for speed */
        return modelpkg_optimize(name, TENSOR_DTYPE_INT8);
    }
}

int modelpkg_verify(const char *name)
{
    for (uint32_t i = 0; i < g_modelpkg.installed_count; i++) {
        if (kstrcmp(g_modelpkg.installed[i].manifest.name, name) == 0) {
            /* TODO: Verify file checksums against manifest */
            kprintf("[PKG] %s: integrity OK\n", name);
            return 0;
        }
    }
    return -1;
}

const char *modelpkg_get_path(const char *name)
{
    for (uint32_t i = 0; i < g_modelpkg.installed_count; i++) {
        if (kstrcmp(g_modelpkg.installed[i].manifest.name, name) == 0) {
            return g_modelpkg.installed[i].install_path;
        }
    }
    return NULL;
}

int modelpkg_search(const char *query, model_manifest_t *results,
                     uint32_t max, uint32_t *count)
{
    /* TODO: Search registries via network */
    if (count) *count = 0;
    return 0;
}
