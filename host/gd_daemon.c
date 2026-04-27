/*
 * gd_daemon.c — Geodessical persistent daemon client
 *
 * Implements the three operations declared in gd_daemon.h:
 *   gd_daemon_ping()      - TCP connect probe (fast, no HTTP)
 *   gd_daemon_spawn()     - CreateProcess + poll until /v1/version responds
 *   gd_daemon_generate()  - POST /v1/generate, parse JSON response
 *
 * Windows-only implementation (matches the geodessical host target).
 */
#define _CRT_SECURE_NO_WARNINGS
#include "gd_daemon.h"

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <sys/socket.h>
#  include <sys/select.h>
#  include <sys/time.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <fcntl.h>
#  define closesocket close
#  define SOCKET int
#  define INVALID_SOCKET (-1)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Portable TCP helpers ────────────────────────────────────────────────── */

static void gd_wsa_init(void) {
#ifdef _WIN32
    static int done = 0;
    if (!done) { WSADATA w; WSAStartup(MAKEWORD(2,2), &w); done = 1; }
#endif
}

/* Try to connect to localhost:port; returns 1 if a server is listening */
int gd_daemon_ping(int port) {
    gd_wsa_init();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return 0;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    /* Short connect timeout via non-blocking mode */
#ifdef _WIN32
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
#else
    int fl = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif
    connect(s, (struct sockaddr *)&addr, sizeof(addr));

    fd_set wset; FD_ZERO(&wset); FD_SET(s, &wset);
    struct timeval tv = { 0, 300000 }; /* 300 ms */
    int r = select((int)s + 1, NULL, &wset, NULL, &tv);
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return r > 0 ? 1 : 0;
}

/* ── HTTP POST helper ────────────────────────────────────────────────────── */

/*
 * Sends a raw HTTP/1.1 request to localhost:port, reads the response body
 * (up to body_cap bytes), returns HTTP status code or -1 on error.
 */
static int gd_http_post(int port, const char *path,
                        const char *body, int body_len,
                        char *resp_buf, int resp_cap)
{
    gd_wsa_init();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        closesocket(s); return -1;
    }

    /* Build request */
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "POST %s HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        path, port, body_len);

    send(s, header, hlen, 0);
    if (body && body_len > 0) send(s, body, body_len, 0);

    /* Read response */
    int total = 0, status = -1;
    char tmp[4096];
    while (1) {
        int n = (int)recv(s, tmp, sizeof(tmp)-1, 0);
        if (n <= 0) break;
        if (total + n < resp_cap - 1) {
            memcpy(resp_buf + total, tmp, (size_t)n);
            total += n;
        }
    }
    resp_buf[total] = '\0';
    closesocket(s);

    /* Parse status line "HTTP/1.x NNN ..." */
    const char *sp = strstr(resp_buf, "HTTP/");
    if (sp) {
        sp = strchr(sp, ' ');
        if (sp) status = atoi(sp + 1);
    }
    return status;
}

/* Same but GET */
static int gd_http_get(int port, const char *path,
                       char *resp_buf, int resp_cap)
{
    gd_wsa_init();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        closesocket(s); return -1;
    }

    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "GET %s HTTP/1.1\r\nHost: localhost:%d\r\nConnection: close\r\n\r\n",
        path, port);
    send(s, hdr, hlen, 0);

    int total = 0;
    char tmp[4096];
    while (1) {
        int n = (int)recv(s, tmp, sizeof(tmp)-1, 0);
        if (n <= 0) break;
        if (total + n < resp_cap - 1) {
            memcpy(resp_buf + total, tmp, (size_t)n);
            total += n;
        }
    }
    resp_buf[total] = '\0';
    closesocket(s);

    int status = -1;
    const char *sp = strstr(resp_buf, "HTTP/");
    if (sp) { sp = strchr(sp, ' '); if (sp) status = atoi(sp + 1); }
    return status;
}

/* ── Minimal JSON field extractor ────────────────────────────────────────── */

/*
 * Finds the value of a JSON field as a null-terminated string in out_buf.
 * Handles string, number, and boolean values.  Not a full parser — only
 * traverses the top-level object.
 */
static int json_get_field(const char *json, const char *key,
                          char *out_buf, int out_cap)
{
    /* Find  "key": in the JSON body (skip HTTP headers) */
    const char *body = strstr(json, "\r\n\r\n");
    if (!body) body = json;
    else body += 4;

    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(body, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == '\t') p++;

    int len = 0;
    if (*p == '"') {
        /* String value — copy until closing quote, handle \" escape */
        p++;
        while (*p && *p != '"' && len < out_cap - 1) {
            if (*p == '\\' && *(p+1) == '"') { out_buf[len++] = '"'; p += 2; continue; }
            if (*p == '\\' && *(p+1) == 'n') { out_buf[len++] = '\n'; p += 2; continue; }
            if (*p == '\\' && *(p+1) == 't') { out_buf[len++] = '\t'; p += 2; continue; }
            if (*p == '\\') { p++; if (*p) { out_buf[len++] = *p++; } continue; }
            out_buf[len++] = *p++;
        }
    } else {
        /* Number / bool / null — copy until delimiter */
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != '\r' && *p != '\n'
               && len < out_cap - 1)
            out_buf[len++] = *p++;
        /* trim trailing whitespace */
        while (len > 0 && (out_buf[len-1] == ' ' || out_buf[len-1] == '\t')) len--;
    }
    out_buf[len] = '\0';
    return len > 0 ? 1 : 0;
}

/* ── Spawn daemon ────────────────────────────────────────────────────────── */

int gd_daemon_spawn(const char *exe_path, const char *model_path,
                    int port, int ctx_size, int timeout_ms)
{
#ifdef _WIN32
    char cmdline[2048];
    if (ctx_size > 0)
        snprintf(cmdline, sizeof(cmdline),
                 "\"%s\" \"%s\" --serve --port %d --ctx-size %d --log-level 1",
                 exe_path, model_path, port, ctx_size);
    else
        snprintf(cmdline, sizeof(cmdline),
                 "\"%s\" \"%s\" --serve --port %d --log-level 1",
                 exe_path, model_path, port);

    STARTUPINFOA si;  memset(&si, 0, sizeof(si)); si.cb = sizeof(si);
    PROCESS_INFORMATION pi; memset(&pi, 0, sizeof(pi));

    /* CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS: survives parent exit */
    if (!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE,
                        CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                        NULL, NULL, &si, &pi)) {
        return 0; /* spawn failed */
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    /* Poll until server responds or timeout */
    int elapsed = 0;
    const int poll_interval = 300; /* ms */
    char buf[1024];
    while (elapsed < timeout_ms) {
        Sleep((DWORD)poll_interval);
        elapsed += poll_interval;
        int rc = gd_http_get(port, "/v1/version", buf, sizeof(buf));
        if (rc == 200) return 1;
    }
    return 0; /* timed out */
#else
    (void)exe_path; (void)model_path; (void)port;
    (void)ctx_size; (void)timeout_ms;
    return 0; /* non-Windows not implemented */
#endif
}

/* ── Generate via daemon ─────────────────────────────────────────────────── */

gd_daemon_result_t gd_daemon_generate(
    const char *exe_path, const char *model_path,
    const char *prompt, int max_tokens,
    float temp, int top_k, float top_p,
    int port, int ctx_size, int no_think)
{
    gd_daemon_result_t res;
    memset(&res, 0, sizeof(res));

    /* DEBUG TRAP: this function has no legitimate call sites in application code.
     * Reaching here during calibration means something corrupted control flow.
     * Print caller addresses and abort to identify the source. */
#ifdef GEODESSICAL_HOSTED
    {
        void *ra0 = __builtin_return_address(0);
        fprintf(stderr,
            "[FATAL] gd_daemon_generate entered unexpectedly! ra0=%p\n"
            "  exe=%s model=%s prompt=%.32s port=%d\n",
            ra0, exe_path ? exe_path : "NULL",
            model_path ? model_path : "NULL",
            prompt ? prompt : "NULL", port);
#ifdef __linux__
        /* Print full backtrace via execve("/proc/self/exe", ...) or just dump addrs */
        void *bt[16];
        int n = 0;
        /* manual unwind: ra0=caller, ra1=caller's caller */
        void *p = __builtin_frame_address(0);
        for (int _i = 0; _i < 16 && p; _i++) {
            void **fp = (void **)p;
            if (!fp[0] || fp[0] == p) break;
            bt[n++] = fp[1];
            p = fp[0];
        }
        fprintf(stderr, "[FATAL] Stack frames:");
        for (int _i = 0; _i < n; _i++) fprintf(stderr, " %p", bt[_i]);
        fprintf(stderr, "\n");
#endif
        fflush(stderr);
        abort();
    }
#endif

    /* 1. Ensure daemon is running */
    if (!gd_daemon_ping(port)) {
        fprintf(stderr, "[GD] No server on port %d — starting daemon...\n", port);
        if (!gd_daemon_spawn(exe_path, model_path, port, ctx_size,
                             GD_DAEMON_STARTUP_TIMEOUT_MS)) {
            snprintf(res.error, sizeof(res.error),
                     "daemon startup timed out (>%d ms)", GD_DAEMON_STARTUP_TIMEOUT_MS);
            return res;
        }
        fprintf(stderr, "[GD] Daemon ready.\n");
    }

    /* 2. Escape prompt for JSON (handle quotes, backslashes, newlines) */
    char esc_prompt[16384];
    {
        int pi = 0, ei = 0;
        int plen = (int)strlen(prompt);
        while (pi < plen && ei < (int)sizeof(esc_prompt) - 4) {
            char c = prompt[pi++];
            if      (c == '"')  { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = '"'; }
            else if (c == '\\') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = '\\'; }
            else if (c == '\n') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = 'n'; }
            else if (c == '\r') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = 'r'; }
            else if (c == '\t') { esc_prompt[ei++] = '\\'; esc_prompt[ei++] = 't'; }
            else                { esc_prompt[ei++] = c; }
        }
        esc_prompt[ei] = '\0';
    }

    /* 3. Build JSON body */
    char body[16896];
    int blen = snprintf(body, sizeof(body),
        "{\"prompt\":\"%s\",\"max_tokens\":%d,\"temperature\":%.3f"
        ",\"top_k\":%d,\"top_p\":%.3f,\"stream\":false}",
        esc_prompt, max_tokens, (double)temp, top_k, (double)top_p);

    /* 4. Send request */
    static char response[131072]; /* 128 KB response buffer */
    int status = gd_http_post(port, "/v1/generate", body, blen,
                              response, (int)sizeof(response));
    if (status != 200) {
        snprintf(res.error, sizeof(res.error),
                 "HTTP %d from /v1/generate", status);
        return res;
    }

    /* 5. Parse response JSON fields */
    char tmp[65536];
    if (json_get_field(response, "text", tmp, sizeof(tmp)))
        strncpy(res.text, tmp, sizeof(res.text) - 1);
    else if (json_get_field(response, "response", tmp, sizeof(tmp)))
        strncpy(res.text, tmp, sizeof(res.text) - 1);

    if (json_get_field(response, "tokens_generated", tmp, sizeof(tmp)))
        res.tokens_generated = atoi(tmp);
    else if (json_get_field(response, "eval_count", tmp, sizeof(tmp)))
        res.tokens_generated = atoi(tmp);

    if (json_get_field(response, "decode_tok_per_s", tmp, sizeof(tmp)))
        res.decode_tok_per_s = atof(tmp);

    if (json_get_field(response, "prefill_ms", tmp, sizeof(tmp)))
        res.prefill_ms = atof(tmp);

    if (json_get_field(response, "total_ms", tmp, sizeof(tmp)))
        res.total_ms = atof(tmp);

    /* Strip <think>...</think> if no_think == 1 */
    if (no_think == 1) {
        char *ts = strstr(res.text, "<think>");
        char *te = ts ? strstr(ts, "</think>") : NULL;
        if (ts && te) {
            te += strlen("</think>");
            while (*te == '\n' || *te == '\r' || *te == ' ') te++;
            memmove(ts, te, strlen(te) + 1);
        }
    }

    res.ok = 1;
    return res;
}
