/*
 * Geodessical Model Auto-Download — Implementation
 *
 * Downloads GGUF models from HuggingFace Hub using WinHTTP.
 * Supports HTTPS, progress reporting, and local caching.
 */

#include "runtime/nn/hf_download.h"

#ifdef GEODESSICAL_HOSTED

#include "hal.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * String helpers
 * ════════════════════════════════════════════════════════════════════════ */

static int str_ends_with(const char *s, const char *suffix) {
    int slen = (int)strlen(s);
    int suflen = (int)strlen(suffix);
    if (suflen > slen) return 0;
    return strcmp(s + slen - suflen, suffix) == 0;
}

static int str_contains_ci(const char *haystack, const char *needle) {
    while (*haystack) {
        const char *h = haystack, *n = needle;
        while (*h && *n) {
            char a = (*h >= 'A' && *h <= 'Z') ? *h + 32 : *h;
            char b = (*n >= 'A' && *n <= 'Z') ? *n + 32 : *n;
            if (a != b) break;
            h++; n++;
        }
        if (!*n) return 1;
        haystack++;
    }
    return 0;
}

static void build_url(char *url, int url_size,
                      const char *repo_id, const char *filename) {
    snprintf(url, url_size,
             "https://huggingface.co/%s/resolve/main/%s",
             repo_id, filename);
}

static void build_api_url(char *url, int url_size,
                          const char *repo_id) {
    snprintf(url, url_size,
             "https://huggingface.co/api/models/%s", repo_id);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════════════ */

void hf_download_init(hf_download_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->status = HF_STATUS_IDLE;
}

void hf_download_set_progress(hf_download_ctx_t *ctx,
                              hf_progress_cb_t cb, void *userdata) {
    if (!ctx) return;
    ctx->progress_cb = cb;
    ctx->progress_ud = userdata;
}

/* ═══════════════════════════════════════════════════════════════════════
 * WinHTTP Download Engine
 * ════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32

/* Convert UTF-8 to wide string */
static wchar_t *utf8_to_wide(const char *s) {
    int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = (wchar_t *)malloc(len * sizeof(wchar_t));
    if (w) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, len);
    return w;
}

static int winhttp_download(hf_download_ctx_t *ctx,
                            const char *url,
                            const char *output_path) {
    HINTERNET hSession = NULL, hConnect = NULL, hRequest = NULL;
    int ret = -1;
    FILE *fp = NULL;

    ctx->status = HF_STATUS_RESOLVING;

    /* Parse URL components */
    wchar_t *wurl = utf8_to_wide(url);
    if (!wurl) { snprintf(ctx->error, sizeof(ctx->error), "Memory allocation failed"); goto done; }

    URL_COMPONENTS uc;
    memset(&uc, 0, sizeof(uc));
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {0}, path[1024] = {0};
    uc.lpszHostName = host;
    uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path;
    uc.dwUrlPathLength = 1024;

    if (!WinHttpCrackUrl(wurl, 0, 0, &uc)) {
        snprintf(ctx->error, sizeof(ctx->error), "Invalid URL: %s", url);
        free(wurl);
        goto done;
    }
    free(wurl);

    /* Open session */
    hSession = WinHttpOpen(L"Geodessical/0.5",
                           WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                           WINHTTP_NO_PROXY_NAME,
                           WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) {
        snprintf(ctx->error, sizeof(ctx->error), "WinHTTP: session open failed");
        goto done;
    }

    /* Connect to host */
    ctx->status = HF_STATUS_CONNECTING;
    hConnect = WinHttpConnect(hSession, host, uc.nPort, 0);
    if (!hConnect) {
        snprintf(ctx->error, sizeof(ctx->error), "WinHTTP: connection failed");
        goto done;
    }

    /* Create request */
    DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    hRequest = WinHttpOpenRequest(hConnect, L"GET", path,
                                  NULL, WINHTTP_NO_REFERER,
                                  WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        snprintf(ctx->error, sizeof(ctx->error), "WinHTTP: request creation failed");
        goto done;
    }

    /* Enable auto-redirect (HF CDN uses 302 redirects) */
    DWORD dwOpt = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hRequest, WINHTTP_OPTION_REDIRECT_POLICY,
                     &dwOpt, sizeof(dwOpt));

    /* Send request */
    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                             WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        snprintf(ctx->error, sizeof(ctx->error), "WinHTTP: send request failed (%lu)",
                 GetLastError());
        goto done;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        snprintf(ctx->error, sizeof(ctx->error), "WinHTTP: no response (%lu)",
                 GetLastError());
        goto done;
    }

    /* Check status code */
    DWORD status_code = 0, sc_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &sc_size,
                        WINHTTP_NO_HEADER_INDEX);
    if (status_code != 200) {
        snprintf(ctx->error, sizeof(ctx->error), "HTTP %lu from %s", status_code, url);
        goto done;
    }

    /* Get content length */
    wchar_t cl_buf[32] = {0};
    DWORD cl_size = sizeof(cl_buf);
    if (WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_CONTENT_LENGTH,
                            WINHTTP_HEADER_NAME_BY_INDEX, cl_buf, &cl_size,
                            WINHTTP_NO_HEADER_INDEX)) {
        ctx->total_bytes = (uint64_t)_wtoi64(cl_buf);
    }

    /* Open output file */
    fp = fopen(output_path, "wb");
    if (!fp) {
        snprintf(ctx->error, sizeof(ctx->error), "Cannot create file: %s", output_path);
        goto done;
    }

    /* Download loop */
    ctx->status = HF_STATUS_DOWNLOADING;
    ctx->downloaded_bytes = 0;
    uint64_t t_start = hal_timer_us();

    DWORD bytes_available = 0;
    uint8_t *chunk = (uint8_t *)malloc(HF_CHUNK_SIZE);
    if (!chunk) {
        snprintf(ctx->error, sizeof(ctx->error), "Memory allocation failed for download buffer");
        goto done;
    }

    for (;;) {
        DWORD bytes_read = 0;
        if (!WinHttpReadData(hRequest, chunk, HF_CHUNK_SIZE, &bytes_read)) {
            snprintf(ctx->error, sizeof(ctx->error),
                     "WinHttpReadData failed at offset %llu (error %lu)",
                     (unsigned long long)ctx->downloaded_bytes,
                     (unsigned long)GetLastError());
            free(chunk);
            goto done;
        }
        if (bytes_read == 0) break;

        if (fwrite(chunk, 1, bytes_read, fp) != bytes_read) {
            snprintf(ctx->error, sizeof(ctx->error), "Write error at offset %llu",
                     (unsigned long long)ctx->downloaded_bytes);
            free(chunk);
            goto done;
        }

        ctx->downloaded_bytes += bytes_read;

        /* Update speed */
        uint64_t elapsed_us = hal_timer_us() - t_start;
        if (elapsed_us > 0) {
            ctx->speed_mbps = (float)ctx->downloaded_bytes /
                              ((float)elapsed_us / 1e6f) / (1024.0f * 1024.0f);
        }

        /* Progress callback */
        if (ctx->progress_cb) {
            ctx->progress_cb(ctx->downloaded_bytes, ctx->total_bytes,
                             ctx->speed_mbps, ctx->progress_ud);
        }
    }

    free(chunk);
    fclose(fp);
    fp = NULL;

    ctx->status = HF_STATUS_COMPLETE;
    ret = 0;

    kprintf("[HF] Download complete: %s (%llu MB, %.1f MB/s)\n",
            output_path,
            (unsigned long long)(ctx->downloaded_bytes / (1024 * 1024)),
            ctx->speed_mbps);

done:
    if (fp) fclose(fp);
    if (hRequest) WinHttpCloseHandle(hRequest);
    if (hConnect) WinHttpCloseHandle(hConnect);
    if (hSession) WinHttpCloseHandle(hSession);

    if (ret != 0) ctx->status = HF_STATUS_ERROR;
    return ret;
}

/* Read JSON response from WinHTTP into buffer */
static int winhttp_get_json(const char *url, char *buf, int buf_size) {
    wchar_t *wurl = utf8_to_wide(url);
    if (!wurl) return -1;

    URL_COMPONENTS uc;
    memset(&uc, 0, sizeof(uc));
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {0}, path[1024] = {0};
    uc.lpszHostName = host;
    uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path;
    uc.dwUrlPathLength = 1024;

    if (!WinHttpCrackUrl(wurl, 0, 0, &uc)) { free(wurl); return -1; }
    free(wurl);

    HINTERNET hSession = WinHttpOpen(L"Geodessical/0.5",
                                     WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return -1;

    HINTERNET hConnect = WinHttpConnect(hSession, host, uc.nPort, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return -1; }

    DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", path,
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return -1;
    }

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                             WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(hRequest, NULL)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return -1;
    }

    int total = 0;
    for (;;) {
        DWORD bytes_read = 0;
        int space = buf_size - total - 1;
        if (space <= 0) break;
        if (!WinHttpReadData(hRequest, buf + total, (DWORD)space, &bytes_read))
            break;
        if (bytes_read == 0) break;
        total += (int)bytes_read;
    }
    buf[total] = '\0';

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return total;
}

#else /* !_WIN32 — Unix: use libcurl or curl CLI as fallback */

static int winhttp_download(hf_download_ctx_t *ctx,
                            const char *url,
                            const char *output_path) {
    /* Fallback: shell out to curl */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "curl -L -o \"%s\" \"%s\"", output_path, url);
    ctx->status = HF_STATUS_DOWNLOADING;
    int rc = system(cmd);
    if (rc == 0) {
        ctx->status = HF_STATUS_COMPLETE;
        return 0;
    }
    snprintf(ctx->error, sizeof(ctx->error), "curl failed (rc=%d)", rc);
    ctx->status = HF_STATUS_ERROR;
    return -1;
}

static int winhttp_get_json(const char *url, char *buf, int buf_size) {
    char cmd[1024];
    char tmpf[] = "/tmp/GD_hf_api.json";
    snprintf(cmd, sizeof(cmd), "curl -sL -o %s \"%s\"", tmpf, url);
    if (system(cmd) != 0) return -1;

    FILE *fp = fopen(tmpf, "r");
    if (!fp) return -1;
    int total = (int)fread(buf, 1, buf_size - 1, fp);
    fclose(fp);
    buf[total] = '\0';
    return total;
}

#endif /* _WIN32 */

/* ═══════════════════════════════════════════════════════════════════════
 * Minimal JSON Parser (for HF API responses)
 *
 * Extracts filenames and sizes from the HF API siblings array.
 * We don't need a full JSON parser — just look for the patterns.
 * ════════════════════════════════════════════════════════════════════════ */

static int parse_hf_siblings(const char *json, hf_file_entry_t *files, int max_files) {
    int count = 0;
    const char *p = json;

    /* Look for "siblings" array */
    const char *siblings = strstr(p, "\"siblings\"");
    if (!siblings) return 0;
    p = siblings;

    /* Find each "rfilename" entry */
    while ((p = strstr(p, "\"rfilename\"")) != NULL && count < max_files) {
        p += 11; /* skip "rfilename" */

        /* Skip to the value string */
        while (*p && *p != '"') p++;
        if (*p != '"') break;
        p++; /* skip opening quote */

        /* Extract filename */
        char *dst = files[count].filename;
        int i = 0;
        while (*p && *p != '"' && i < HF_MAX_FILENAME - 1) {
            dst[i++] = *p++;
        }
        dst[i] = '\0';

        /* Check if it's a GGUF file */
        files[count].is_gguf = str_ends_with(dst, ".gguf");
        files[count].size = 0; /* Size not always in siblings response */

        count++;
    }

    return count;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════════════ */

int hf_list_gguf_files(hf_download_ctx_t *ctx, const char *repo_id) {
    if (!ctx || !repo_id) return -1;

    strncpy(ctx->repo_id, repo_id, HF_MAX_REPO_ID - 1);

    char api_url[HF_MAX_URL];
    build_api_url(api_url, sizeof(api_url), repo_id);

    /* Fetch API response */
    char *json = (char *)malloc(256 * 1024); /* 256 KB for API response */
    if (!json) {
        snprintf(ctx->error, sizeof(ctx->error), "Memory allocation failed");
        return -1;
    }

    kprintf("[HF] Querying repo: %s\n", repo_id);
    int json_len = winhttp_get_json(api_url, json, 256 * 1024);
    if (json_len <= 0) {
        snprintf(ctx->error, sizeof(ctx->error), "Failed to fetch API for %s", repo_id);
        free(json);
        return -1;
    }

    /* Parse file listing */
    int total = parse_hf_siblings(json, ctx->files, HF_MAX_FILES);
    free(json);

    /* Filter to GGUF files only */
    ctx->n_files = 0;
    for (int i = 0; i < total; i++) {
        if (ctx->files[i].is_gguf) {
            if (i != ctx->n_files)
                ctx->files[ctx->n_files] = ctx->files[i];
            ctx->n_files++;
        }
    }

    kprintf("[HF] Found %d GGUF files in %s\n", ctx->n_files, repo_id);
    for (int i = 0; i < ctx->n_files; i++) {
        kprintf("[HF]   %d. %s\n", i + 1, ctx->files[i].filename);
    }

    return ctx->n_files;
}

int hf_download_file(hf_download_ctx_t *ctx,
                     const char *repo_id,
                     const char *filename,
                     const char *output_dir) {
    if (!ctx || !repo_id || !filename || !output_dir) return -1;

    strncpy(ctx->repo_id, repo_id, HF_MAX_REPO_ID - 1);
    strncpy(ctx->filename, filename, HF_MAX_FILENAME - 1);
    strncpy(ctx->output_dir, output_dir, HF_MAX_PATH - 1);

    /* Build output path */
    snprintf(ctx->output_path, HF_MAX_PATH, "%s/%s", output_dir, filename);

    /* Check if already cached */
    FILE *test = fopen(ctx->output_path, "rb");
    if (test) {
        fseek(test, 0, SEEK_END);
        long sz = ftell(test);
        fclose(test);
        if (sz > 1024) { /* Non-trivial size = likely valid */
            kprintf("[HF] Model already cached: %s (%ld MB)\n",
                    ctx->output_path, sz / (1024 * 1024));
            ctx->status = HF_STATUS_CACHED;
            ctx->total_bytes = (uint64_t)sz;
            ctx->downloaded_bytes = (uint64_t)sz;
            return 0;
        }
    }

    /* Build download URL */
    char url[HF_MAX_URL];
    build_url(url, sizeof(url), repo_id, filename);

    kprintf("[HF] Downloading: %s\n", url);
    kprintf("[HF] Saving to: %s\n", ctx->output_path);

    /* Ensure output directory exists */
#ifdef _WIN32
    CreateDirectoryA(output_dir, NULL);
#else
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", output_dir);
    system(mkdir_cmd);
#endif

    return winhttp_download(ctx, url, ctx->output_path);
}

int hf_download_auto(hf_download_ctx_t *ctx,
                     const char *repo_id,
                     const char *quant_hint,
                     const char *output_dir) {
    if (!ctx || !repo_id || !quant_hint) return -1;

    /* List files in the repo */
    int n = hf_list_gguf_files(ctx, repo_id);
    if (n <= 0) {
        snprintf(ctx->error, sizeof(ctx->error),
                 "No GGUF files found in %s", repo_id);
        ctx->status = HF_STATUS_ERROR;
        return -1;
    }

    /* Find best match for quant hint */
    int best = -1;
    for (int i = 0; i < ctx->n_files; i++) {
        if (str_contains_ci(ctx->files[i].filename, quant_hint)) {
            best = i;
            break;
        }
    }

    /* Fallback to first GGUF file if no match */
    if (best < 0) {
        kprintf("[HF] No match for '%s', using first GGUF: %s\n",
                quant_hint, ctx->files[0].filename);
        best = 0;
    }

    kprintf("[HF] Selected: %s\n", ctx->files[best].filename);
    return hf_download_file(ctx, repo_id, ctx->files[best].filename, output_dir);
}

int hf_is_cached(const char *repo_id, const char *filename,
                 const char *output_dir, char *out_path, int path_size) {
    (void)repo_id;
    if (!filename || !output_dir) return 0;

    char path[HF_MAX_PATH];
    snprintf(path, sizeof(path), "%s/%s", output_dir, filename);

    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);

    if (sz <= 1024) return 0; /* Too small to be a valid model */

    if (out_path && path_size > 0)
        snprintf(out_path, path_size, "%s", path);

    return 1;
}

hf_status_t hf_download_status(const hf_download_ctx_t *ctx) {
    return ctx ? ctx->status : HF_STATUS_ERROR;
}

const char *hf_download_error(const hf_download_ctx_t *ctx) {
    return ctx ? ctx->error : "null context";
}

void hf_download_free(hf_download_ctx_t *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
}

#endif /* GEODESSICAL_HOSTED */
