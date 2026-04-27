/*
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
 * ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
 * ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
 * ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
 * ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
 * ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
 * :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
 * :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
 * ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
 * :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
 * ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
 * ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
 * :::::::::................................:@@@@@@@@@@%:...............................::::::
 * ::::::::..................................*@@@@@@@@@-................................::::::::
 * ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
 * :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
 * :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
 * :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
 * :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
 * :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
 * :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
 * :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
 * :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
 * :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
 * ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
 * ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
 * :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
 * ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
 * :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
 * :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
 * ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
 * ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
 * :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
 * ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
 * ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
 * :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

/*
 * Geodessical HTTP API Server
 *
 * Native Geodessical REST API for LLM inference.
 * Uses raw Winsock2 — no external dependencies.
 *
 * Continuous-batching model:
 *   - A dedicated inference worker thread drains a bounded request queue.
 *   - The TCP accept loop spawns a lightweight handler thread per connection
 *     that parses the request, enqueues it, and streams the response back
 *     once the worker has produced output.
 *   - This serialises model execution (one inference at a time) while letting
 *     multiple clients connect and receive responses without TCP timeout.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "api_server.h"
#include "hal.h"
#include "../runtime/nn/llm.h"
#include "web_ui.h"
#include "mcp_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <process.h>   /* _beginthreadex */
#  pragma comment(lib, "ws2_32")
typedef SOCKET socket_t;
#define SOCKET_INVALID INVALID_SOCKET
#define sock_close closesocket
typedef HANDLE thread_t;
typedef CRITICAL_SECTION mutex_t;
typedef CONDITION_VARIABLE condvar_t;
#define mutex_init(m)   InitializeCriticalSection(m)
#define mutex_lock(m)   EnterCriticalSection(m)
#define mutex_unlock(m) LeaveCriticalSection(m)
#define mutex_destroy(m) DeleteCriticalSection(m)
#define condvar_init(c)   InitializeConditionVariable(c)
#define condvar_wait(c,m) SleepConditionVariableCS(c, m, INFINITE)
#define condvar_signal(c) WakeConditionVariable(c)
#define condvar_broadcast(c) WakeAllConditionVariable(c)
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
#  include <pthread.h>
typedef int socket_t;
#define SOCKET_INVALID (-1)
#define sock_close close
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t condvar_t;
#define mutex_init(m)   pthread_mutex_init(m, NULL)
#define mutex_lock(m)   pthread_mutex_lock(m)
#define mutex_unlock(m) pthread_mutex_unlock(m)
#define mutex_destroy(m) pthread_mutex_destroy(m)
#define condvar_init(c)   pthread_cond_init(c, NULL)
#define condvar_wait(c,m) pthread_cond_wait(c, m)
#define condvar_signal(c) pthread_cond_signal(c)
#define condvar_broadcast(c) pthread_cond_broadcast(c)
#endif

/* ─── State ─── */
static volatile int g_running = 1;
static socket_t g_listen_sock = SOCKET_INVALID;

/* ─── Inference serialisation mutex ─────────────────────────────────────────
 * All inference calls (llm_prompt_n, llm_chat_turn) must hold this lock.
 * Accepted connections queue their parsed requests, then block on a per-
 * request condition variable until the worker thread signals completion.  */
static mutex_t  g_infer_lock;
static condvar_t g_queue_ready;   /* signalled when a new item is enqueued */
static condvar_t g_result_ready;  /* signalled when current item is done */

#define CBATCH_QUEUE_MAX 32

typedef struct {
    char   prompt[8192];
    char   output[131072];
    int    max_tokens;
    float  temperature;
    int    is_chat;          /* 0 = /v1/generate, 1 = /v1/chat */
    int    use_geodesic;     /* 1 = geodesic speculative decode path */
    int    n_draft;          /* geodesic lookahead depth (0 = auto) */
    int    done;             /* set to 1 by worker when complete */
    int    n_generated;      /* token count */
    int    error;            /* 1 on error */
    int    stream;           /* 1 = SSE streaming: worker sends tokens to stream_sock */
    socket_t stream_sock;    /* valid when stream=1; worker writes SSE events here */
    float  decode_tok_per_s; /* filled by worker: decode throughput */
    float  prefill_ms;       /* filled by worker: prefill latency */
    float  total_ms;         /* filled by worker: end-to-end latency */
} cbatch_request_t;

static cbatch_request_t  g_queue_slots[CBATCH_QUEUE_MAX];
static int               g_queue_head = 0;   /* next slot to dequeue */
static int               g_queue_tail = 0;   /* next slot to enqueue */
static int               g_queue_count = 0;

/* Enqueue a new request.  Returns pointer to the slot (held under lock),
 * or NULL if the queue is full. */
static cbatch_request_t *cbatch_enqueue(void)
{
    if (g_queue_count >= CBATCH_QUEUE_MAX) return NULL;
    cbatch_request_t *slot = &g_queue_slots[g_queue_tail];
    memset(slot, 0, sizeof(*slot));
    g_queue_tail = (g_queue_tail + 1) % CBATCH_QUEUE_MAX;
    g_queue_count++;
    return slot;
}

/* Dequeue the head item.  Called by worker under lock. */
static cbatch_request_t *cbatch_dequeue(void)
{
    if (g_queue_count == 0) return NULL;
    cbatch_request_t *slot = &g_queue_slots[g_queue_head];
    g_queue_head = (g_queue_head + 1) % CBATCH_QUEUE_MAX;
    g_queue_count--;
    return slot;
}

/* Forward declarations for SSE helpers (defined after send_json below) */
static void sse_send_token(socket_t sock, const char *text, int n);
static void sse_send_done(socket_t sock, int n_generated);

/* ─── Inference worker thread ────────────────────────────────────────────── *
 *
 * Parallel continuous batching:
 *   - Accepts up to PDEC_MAX_ACTIVE requests simultaneously.
 *   - Each active slot gets a session (llm_session_create/step).
 *   - Worker iterates all active sessions, stepping each one token per
 *     iteration, then loops.  When a session completes, the request is
 *     finished and the slot is freed for the next queued request.
 *   - Requests that can't use the session path (chat, geodesic) fall
 *     back to the serial path and block until complete.
 *
 * ─────────────────────────────────────────────────────────────────────────── */
#define PDEC_MAX_ACTIVE 4

typedef struct {
    cbatch_request_t *req;         /* pointer into g_queue_slots */
    int               session_id;  /* llm session slot (-1 = not created) */
    int               tok_buf[4096];
    int               n_tokens;    /* generated token ids so far */
    uint64_t          t_start_us;  /* wall-clock at first token step */
    uint64_t          t_first_us;  /* wall-clock after first token (TTFT) */
    float             prefill_ms;  /* prefill latency captured at creation */
} pdec_active_t;

static pdec_active_t g_pdec[PDEC_MAX_ACTIVE];
static int           g_pdec_n = 0;  /* number of active slots */

#ifdef _WIN32
static unsigned __stdcall infer_worker(void *arg)
#else
static void *infer_worker(void *arg)
#endif
{
    (void)arg;

    /* Initialise parallel decode slots */
    for (int i = 0; i < PDEC_MAX_ACTIVE; i++) {
        g_pdec[i].req        = NULL;
        g_pdec[i].session_id = -1;
        g_pdec[i].n_tokens   = 0;
        g_pdec[i].t_start_us = 0;
        g_pdec[i].t_first_us = 0;
        g_pdec[i].prefill_ms = 0.0f;
    }
    g_pdec_n = 0;

    mutex_lock(&g_infer_lock);
    while (g_running) {

        /* ── Admit new requests into free parallel-decode slots ─────────── */
        while (g_queue_count > 0 && g_pdec_n < PDEC_MAX_ACTIVE) {
            cbatch_request_t *req = cbatch_dequeue();
            if (!req) break;

            /* Chat and geodesic requests must run serially (they manage their
             * own context; can't be interleaved at token granularity). */
            if (req->is_chat || req->use_geodesic) {
                /* Release lock, run, re-acquire. */
                mutex_unlock(&g_infer_lock);

                if (req->use_geodesic) {
                    static int geo_tok_buf[4096];
                    static int geo_out_buf[4096];
                    int n_ptok = llm_tokenize_text(req->prompt, geo_tok_buf, 4095);
                    int nd = req->n_draft > 0 ? req->n_draft : 4;
                    if (n_ptok > 0) {
                        int n = llm_generate_geodesic_speculative(
                                    geo_tok_buf, n_ptok,
                                    geo_out_buf, 4095,
                                    req->max_tokens, req->temperature, nd);
                        if (n > 0)
                            req->n_generated = llm_decode_tokens(geo_out_buf, n,
                                                   req->output, (int)sizeof(req->output));
                        else req->error = 1;
                    } else req->error = 1;
                } else {
                    /* Chat: optionally stream tokens via llm_set_stream_cb */
                    if (req->stream) {
                        llm_set_stream_cb(
                            (llm_token_cb_t)(void(*)(const char*,int,void*))sse_send_token,
                            (void *)(intptr_t)req->stream_sock);
                    }
                    req->n_generated = llm_chat_turn(req->prompt, req->output,
                                                     (int)sizeof(req->output),
                                                     req->max_tokens,
                                                     req->temperature);
                    if (req->stream) {
                        llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
                        sse_send_done(req->stream_sock, req->n_generated);
                    }
                }

                mutex_lock(&g_infer_lock);
                req->done = 1;
                condvar_broadcast(&g_result_ready);
                continue;
            }

            /* Create a session for this request */
            mutex_unlock(&g_infer_lock);

            int tok_buf[4096];
            int n_ptok = llm_tokenize_text(req->prompt, tok_buf, 4095);
            int sid = -1;
            if (n_ptok > 0)
                sid = llm_session_create(tok_buf, n_ptok,
                                         req->max_tokens, req->temperature);

            mutex_lock(&g_infer_lock);

            if (sid < 0) {
                /* Failed to create session (model not loaded, all slots busy) */
                req->error = 1;
                req->done  = 1;
                condvar_broadcast(&g_result_ready);
                continue;
            }

            /* Find a free pdec slot */
            for (int i = 0; i < PDEC_MAX_ACTIVE; i++) {
                if (!g_pdec[i].req) {
                    g_pdec[i].req        = req;
                    g_pdec[i].session_id = sid;
                    g_pdec[i].n_tokens   = 0;
                    g_pdec[i].t_start_us = hal_timer_us();
                    g_pdec[i].t_first_us = 0;
                    g_pdec[i].prefill_ms = llm_last_prefill_ms();
                    g_pdec_n++;
                    break;
                }
            }
        }

        /* If no active sessions and no queued requests, wait */
        if (g_pdec_n == 0) {
            while (g_queue_count == 0 && g_running)
                condvar_wait(&g_queue_ready, &g_infer_lock);
            continue;
        }

        /* ── Step each active session one token ─────────────────────────── */
        mutex_unlock(&g_infer_lock);

        for (int i = 0; i < PDEC_MAX_ACTIVE; i++) {
            if (!g_pdec[i].req) continue;

            int tok = llm_session_step(g_pdec[i].session_id);

            if (tok == LLM_SESSION_DONE || tok == LLM_SESSION_ERR) {
                /* Session complete: detokenise and finish request */
                cbatch_request_t *req = g_pdec[i].req;
                uint64_t t_done_us = hal_timer_us();
                if (req->stream) {
                    /* SSE: send final done event; output was already streamed */
                    sse_send_done(req->stream_sock, g_pdec[i].n_tokens);
                    req->n_generated = g_pdec[i].n_tokens;
                } else if (g_pdec[i].n_tokens > 0) {
                    req->n_generated = llm_decode_tokens(
                        g_pdec[i].tok_buf, g_pdec[i].n_tokens,
                        req->output, (int)sizeof(req->output));
                }
                if (tok == LLM_SESSION_ERR) req->error = 1;
                /* Compute timing stats */
                {
                    uint64_t gen_us = t_done_us - g_pdec[i].t_start_us;
                    int n = g_pdec[i].n_tokens;
                    req->decode_tok_per_s = (gen_us > 0 && n > 0)
                        ? (float)n * 1000000.0f / (float)gen_us : 0.0f;
                    req->prefill_ms  = g_pdec[i].prefill_ms;
                    req->total_ms    = (float)(t_done_us - g_pdec[i].t_start_us) / 1000.0f
                                       + g_pdec[i].prefill_ms;
                }
                llm_session_destroy(g_pdec[i].session_id);

                mutex_lock(&g_infer_lock);
                req->done = 1;
                condvar_broadcast(&g_result_ready);
                g_pdec[i].req        = NULL;
                g_pdec[i].session_id = -1;
                g_pdec[i].n_tokens   = 0;
                g_pdec[i].t_start_us = 0;
                g_pdec[i].t_first_us = 0;
                g_pdec[i].prefill_ms = 0.0f;
                g_pdec_n--;
                mutex_unlock(&g_infer_lock);
            } else if (tok >= 0) {
                /* Accumulate generated token */
                if (g_pdec[i].n_tokens < 4095)
                    g_pdec[i].tok_buf[g_pdec[i].n_tokens++] = tok;
                /* SSE: push token text to client immediately */
                if (g_pdec[i].req->stream) {
                    char tok_text[256];
                    int tlen = llm_decode_tokens(&tok, 1, tok_text, (int)sizeof(tok_text));
                    if (tlen > 0)
                        sse_send_token(g_pdec[i].req->stream_sock, tok_text, tlen);
                }
            }
        }

        mutex_lock(&g_infer_lock);
    }

    /* Cleanup any in-flight sessions on shutdown */
    for (int i = 0; i < PDEC_MAX_ACTIVE; i++) {
        if (g_pdec[i].req) {
            g_pdec[i].req->error = 1;
            g_pdec[i].req->done  = 1;
            llm_session_destroy(g_pdec[i].session_id);
        }
    }
    condvar_broadcast(&g_result_ready);
    mutex_unlock(&g_infer_lock);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ─── Request parsing ─── */
#define MAX_REQ_SIZE  (256 * 1024)  /* 256 KB max request body */
#define MAX_RESP_SIZE (512 * 1024)  /* 512 KB max response */

typedef struct {
    char method[16];       /* GET, POST, etc. */
    char path[256];        /* /api/generate, etc. */
    char *body;            /* Request body (JSON) */
    int   body_len;
    int   content_length;
} http_request_t;

/* ─── Simple JSON builder helpers ─── */
static int json_append(char *buf, int pos, int max, const char *s) {
    while (*s && pos < max - 1) buf[pos++] = *s++;
    return pos;
}

static int json_append_str(char *buf, int pos, int max, const char *key, const char *val) {
    pos = json_append(buf, pos, max, "\"");
    pos = json_append(buf, pos, max, key);
    pos = json_append(buf, pos, max, "\":\"");
    /* Escape special chars in value */
    while (*val && pos < max - 3) {
        if (*val == '"') { buf[pos++] = '\\'; buf[pos++] = '"'; }
        else if (*val == '\\') { buf[pos++] = '\\'; buf[pos++] = '\\'; }
        else if (*val == '\n') { buf[pos++] = '\\'; buf[pos++] = 'n'; }
        else if (*val == '\r') { buf[pos++] = '\\'; buf[pos++] = 'r'; }
        else if (*val == '\t') { buf[pos++] = '\\'; buf[pos++] = 't'; }
        else buf[pos++] = *val;
        val++;
    }
    buf[pos++] = '"';
    return pos;
}

static int json_append_int(char *buf, int pos, int max, const char *key, int val) {
    pos = json_append(buf, pos, max, "\"");
    pos = json_append(buf, pos, max, key);
    pos = json_append(buf, pos, max, "\":");
    char num[32];
    snprintf(num, sizeof(num), "%d", val);
    pos = json_append(buf, pos, max, num);
    return pos;
}

static int json_append_bool(char *buf, int pos, int max, const char *key, int val) {
    pos = json_append(buf, pos, max, "\"");
    pos = json_append(buf, pos, max, key);
    pos = json_append(buf, pos, max, "\":");
    pos = json_append(buf, pos, max, val ? "true" : "false");
    return pos;
}

/* ─── Minimal JSON parser (extract string value by key) ─── */
static int json_find_str(const char *json, const char *key, char *out, int max) {
    /* Find "key": "value" pattern */
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return -1;
    p++; /* skip opening quote */
    int i = 0;
    while (*p && *p != '"' && i < max - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            if (*p == 'n') out[i++] = '\n';
            else if (*p == 't') out[i++] = '\t';
            else if (*p == 'r') out[i++] = '\r';
            else out[i++] = *p;
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return i;
}

static int json_find_int(const char *json, const char *key, int default_val) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return default_val;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return atoi(p);
}

static float json_find_float(const char *json, const char *key, float default_val) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return default_val;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return (float)atof(p);
}

/* ─── HTTP response helpers ─── */
static void send_response(socket_t sock, int status, const char *content_type,
                          const char *body, int body_len) {
    const char *status_str = "200 OK";
    if (status == 404) status_str = "404 Not Found";
    else if (status == 400) status_str = "400 Bad Request";
    else if (status == 405) status_str = "405 Method Not Allowed";
    else if (status == 500) status_str = "500 Internal Server Error";

    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_str, content_type, body_len);

    send(sock, header, hlen, 0);
    if (body && body_len > 0)
        send(sock, body, body_len, 0);
}

static void send_json(socket_t sock, int status, const char *json) {
    send_response(sock, status, "application/json", json, (int)strlen(json));
}

/* ─── SSE helpers ─── */
static void sse_send_headers(socket_t sock) {
    const char *hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n";
    send(sock, hdr, (int)strlen(hdr), 0);
}

static void sse_send_token(socket_t sock, const char *text, int n) {
    char buf[640];
    int p = 0;
    p += snprintf(buf + p, (int)sizeof(buf) - p, "data: {\"response\":\"");
    for (int i = 0; i < n && p < (int)sizeof(buf) - 24; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '"')       { buf[p++] = '\\'; buf[p++] = '"';  }
        else if (c == '\\') { buf[p++] = '\\'; buf[p++] = '\\'; }
        else if (c == '\n') { buf[p++] = '\\'; buf[p++] = 'n';  }
        else if (c == '\r') { buf[p++] = '\\'; buf[p++] = 'r';  }
        else                  buf[p++] = (char)c;
    }
    p += snprintf(buf + p, (int)sizeof(buf) - p, "\",\"done\":false}\n\n");
    send(sock, buf, p, 0);
}

static void sse_send_done(socket_t sock, int n_generated) {
    char buf[128];
    int p = snprintf(buf, (int)sizeof(buf),
        "data: {\"response\":\"\",\"done\":true,\"eval_count\":%d}\n\n",
        n_generated);
    send(sock, buf, p, 0);
}

/* ─── Parse HTTP request ─── */
static int parse_request(socket_t sock, http_request_t *req) {
    static char buf[MAX_REQ_SIZE];
    int total = 0;

    /* Read headers */
    while (total < MAX_REQ_SIZE - 1) {
        int n = recv(sock, buf + total, MAX_REQ_SIZE - 1 - total, 0);
        if (n <= 0) return -1;
        total += n;
        buf[total] = '\0';

        /* Check for end of headers */
        char *hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            /* Parse method and path */
            char *sp1 = strchr(buf, ' ');
            if (!sp1) return -1;
            int mlen = (int)(sp1 - buf);
            if (mlen >= (int)sizeof(req->method)) mlen = sizeof(req->method) - 1;
            memcpy(req->method, buf, mlen);
            req->method[mlen] = '\0';

            char *sp2 = strchr(sp1 + 1, ' ');
            if (!sp2) return -1;
            int plen = (int)(sp2 - sp1 - 1);
            if (plen >= (int)sizeof(req->path)) plen = sizeof(req->path) - 1;
            memcpy(req->path, sp1 + 1, plen);
            req->path[plen] = '\0';

            /* Find Content-Length */
            req->content_length = 0;
            const char *cl = strstr(buf, "Content-Length:");
            if (!cl) cl = strstr(buf, "content-length:");
            if (cl) req->content_length = atoi(cl + 15);

            /* Body starts after \r\n\r\n */
            int hdr_size = (int)(hdr_end + 4 - buf);
            int body_received = total - hdr_size;
            req->body = buf + hdr_size;

            /* Read remaining body if needed */
            while (body_received < req->content_length &&
                   total < MAX_REQ_SIZE - 1) {
                n = recv(sock, buf + total, MAX_REQ_SIZE - 1 - total, 0);
                if (n <= 0) break;
                total += n;
                body_received += n;
            }
            buf[total] = '\0';
            req->body_len = body_received;
            return 0;
        }
    }
    return -1;
}

/* ─── API Handlers ─── */

static void handle_version(socket_t sock) {
    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"name\":\"Geodessical\",\"version\":\"0.5.0\",\"backend\":\"%s\"}",
             llm_backend_name());
    send_json(sock, 200, resp);
}

static void handle_health(socket_t sock) {
    send_json(sock, 200, "{\"status\":\"ok\"}");
}

static void handle_tags(socket_t sock) {
    char resp[2048];
    int p = 0;
    p = json_append(resp, p, sizeof(resp), "{");
    p = json_append_str(resp, p, sizeof(resp), "model", llm_model_name());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_str(resp, p, sizeof(resp), "arch", llm_model_arch());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "layers", llm_model_layers());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "dim", llm_model_dim());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "vocab", llm_model_vocab());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_str(resp, p, sizeof(resp), "backend", llm_backend_name());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "context_tokens", llm_chat_context_tokens());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "context_max", llm_chat_context_max());
    int vram = llm_last_vram_usage_mb();
    if (vram > 0) {
        p = json_append(resp, p, sizeof(resp), ",");
        p = json_append_int(resp, p, sizeof(resp), "vram_mb", vram);
    }
    p = json_append(resp, p, sizeof(resp), "}");
    resp[p] = '\0';
    send_json(sock, 200, resp);
}

static void handle_logprobs(socket_t sock, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        send_json(sock, 400, "{\"error\":\"empty request body\"}");
        return;
    }
    char prompt[8192];
    if (json_find_str(req->body, "prompt", prompt, sizeof(prompt)) < 0) {
        send_json(sock, 400, "{\"error\":\"missing 'prompt' field\"}");
        return;
    }
    int max_tokens = json_find_int(req->body, "max_tokens", 512);
    if (max_tokens < 2)   max_tokens = 2;
    if (max_tokens > 4096) max_tokens = 4096;

    float *lps = (float *)malloc((size_t)max_tokens * sizeof(float));
    if (!lps) { send_json(sock, 500, "{\"error\":\"OOM\"}"); return; }

    mutex_lock(&g_infer_lock);
    int n = llm_eval_sequence_logprobs(prompt, lps, max_tokens);
    mutex_unlock(&g_infer_lock);

    if (n < 0) {
        free(lps);
        send_json(sock, 503, "{\"error\":\"inference failed\"}");
        return;
    }

    /* Build JSON: {"logprobs":[lp0,lp1,...],"n_tokens":N} */
    char *resp = (char *)malloc(32 + (size_t)n * 20);
    if (!resp) { free(lps); send_json(sock, 500, "{\"error\":\"OOM\"}"); return; }
    int p = 0;
    p += sprintf(resp + p, "{\"logprobs\":[");
    for (int i = 0; i < n; i++) {
        if (i > 0) resp[p++] = ',';
        p += sprintf(resp + p, "%.6f", (double)lps[i]);
    }
    p += sprintf(resp + p, "],\"n_tokens\":%d}", n);
    free(lps);
    send_json(sock, 200, resp);
    free(resp);
}

static void handle_generate(socket_t sock, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        send_json(sock, 400, "{\"error\":\"empty request body\"}");
        return;
    }

    char prompt[8192];
    if (json_find_str(req->body, "prompt", prompt, sizeof(prompt)) < 0) {
        send_json(sock, 400, "{\"error\":\"missing 'prompt' field\"}");
        return;
    }

    int max_tokens = json_find_int(req->body, "num_predict", 128);
    if (max_tokens > 4096) max_tokens = 4096;
    float temp = json_find_float(req->body, "temperature", 0.7f);

    /* Optional geodesic mode: "mode": "geodesic" and "n_draft": N */
    char mode_buf[32] = {0};
    int use_geodesic = 0;
    int n_draft = 4;
    if (json_find_str(req->body, "mode", mode_buf, sizeof(mode_buf)) >= 0 &&
        strncmp(mode_buf, "geodesic", 8) == 0) {
        use_geodesic = 1;
        n_draft = json_find_int(req->body, "n_draft", 4);
        if (n_draft < 1) n_draft = 1;
        if (n_draft > 8) n_draft = 8;
    }

    int do_stream = json_find_int(req->body, "stream", 0);
    if (do_stream) sse_send_headers(sock);

    /* Enqueue the request and wait for the inference worker */
    mutex_lock(&g_infer_lock);
    cbatch_request_t *slot = cbatch_enqueue();
    if (!slot) {
        mutex_unlock(&g_infer_lock);
        if (do_stream)
            sse_send_done(sock, 0);
        else
            send_json(sock, 503, "{\"error\":\"server busy, request queue full\"}");
        return;
    }
    strncpy(slot->prompt, prompt, sizeof(slot->prompt) - 1);
    slot->max_tokens    = max_tokens;
    slot->temperature   = temp;
    slot->is_chat       = 0;
    slot->use_geodesic  = use_geodesic;
    slot->n_draft       = n_draft;
    slot->stream        = do_stream;
    slot->stream_sock   = do_stream ? sock : SOCKET_INVALID;
    slot->done          = 0;
    condvar_signal(&g_queue_ready);

    /* Block until worker signals completion */
    while (!slot->done)
        condvar_wait(&g_result_ready, &g_infer_lock);
    int n = slot->n_generated;
    float dec_tps  = slot->decode_tok_per_s;
    float pf_ms    = slot->prefill_ms;
    float tot_ms   = slot->total_ms;
    char output[131072];
    memcpy(output, slot->output, sizeof(output));
    mutex_unlock(&g_infer_lock);

    if (do_stream) return; /* SSE: all data already sent by worker */

    /* Build response JSON */
    char resp[MAX_RESP_SIZE];
    int p = 0;
    p = json_append(resp, p, sizeof(resp), "{");
    p = json_append_str(resp, p, sizeof(resp), "model", llm_model_name());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_str(resp, p, sizeof(resp), "response", output);
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_bool(resp, p, sizeof(resp), "done", 1);
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "eval_count", n > 0 ? n : 0);
    {
        char tbuf[192];
        snprintf(tbuf, sizeof(tbuf),
                 ",\"decode_tok_per_s\":%.1f,\"prefill_ms\":%.0f,\"total_ms\":%.0f",
                 (double)dec_tps, (double)pf_ms, (double)tot_ms);
        if (p + (int)strlen(tbuf) < (int)sizeof(resp) - 2)
            p += snprintf(resp + p, sizeof(resp) - (size_t)p, "%s", tbuf);
    }
    p = json_append(resp, p, sizeof(resp), "}");
    resp[p] = '\0';

    send_json(sock, 200, resp);
}

static void handle_chat(socket_t sock, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        send_json(sock, 400, "{\"error\":\"empty request body\"}");
        return;
    }

    char content[8192];
    content[0] = '\0';

    /* Try simple format first: {"prompt": "..."} */
    if (json_find_str(req->body, "prompt", content, sizeof(content)) < 0) {
        /* Try messages array format: find the last "role":"user" */
        const char *last_user = (const char *)0;
        const char *p = req->body;
        while ((p = strstr(p, "\"user\"")) != (const char *)0) {
            last_user = p;
            p += 6;
        }
        if (last_user) {
            const char *c = strstr(last_user, "\"content\"");
            if (c) {
                c += 9;
                while (*c == ' ' || *c == ':') c++;
                if (*c == '"') {
                    c++;
                    int i = 0;
                    while (*c && *c != '"' && i < (int)sizeof(content) - 1) {
                        if (*c == '\\' && *(c+1)) {
                            c++;
                            if (*c == 'n') content[i++] = '\n';
                            else if (*c == 't') content[i++] = '\t';
                            else content[i++] = *c;
                        } else {
                            content[i++] = *c;
                        }
                        c++;
                    }
                    content[i] = '\0';
                }
            }
        }
    }

    if (content[0] == '\0') {
        send_json(sock, 400, "{\"error\":\"no user message found\"}");
        return;
    }

    int max_tokens = json_find_int(req->body, "max_tokens",
                    json_find_int(req->body, "num_predict", 128));
    if (max_tokens > 4096) max_tokens = 4096;
    float temp = json_find_float(req->body, "temperature", 0.7f);

    int do_stream = json_find_int(req->body, "stream", 0);
    if (do_stream) sse_send_headers(sock);

    /* Enqueue the request and wait for the inference worker */
    mutex_lock(&g_infer_lock);
    cbatch_request_t *slot = cbatch_enqueue();
    if (!slot) {
        mutex_unlock(&g_infer_lock);
        if (do_stream)
            sse_send_done(sock, 0);
        else
            send_json(sock, 503, "{\"error\":\"server busy, request queue full\"}");
        return;
    }
    strncpy(slot->prompt, content, sizeof(slot->prompt) - 1);
    slot->max_tokens  = max_tokens;
    slot->temperature = temp;
    slot->is_chat     = 1;
    slot->stream      = do_stream;
    slot->stream_sock = do_stream ? sock : SOCKET_INVALID;
    slot->done        = 0;
    condvar_signal(&g_queue_ready);

    while (!slot->done)
        condvar_wait(&g_result_ready, &g_infer_lock);
    int n = slot->n_generated;
    char output[131072];
    memcpy(output, slot->output, sizeof(output));
    mutex_unlock(&g_infer_lock);

    if (do_stream) return; /* SSE: all data already sent by worker */

    float tps = llm_last_tok_per_sec();
    float elapsed_ms = (tps > 0.0f && n > 0) ? (float)n * 1000.0f / tps : 0;

    /* Build response JSON — compatible with both our UI and Ollama format */
    char resp[MAX_RESP_SIZE];
    int rp = 0;
    rp = json_append(resp, rp, sizeof(resp), "{");
    rp = json_append_str(resp, rp, sizeof(resp), "model", llm_model_name());
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_str(resp, rp, sizeof(resp), "response", output);
    rp = json_append(resp, rp, sizeof(resp), ",\"message\":{");
    rp = json_append_str(resp, rp, sizeof(resp), "role", "assistant");
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_str(resp, rp, sizeof(resp), "content", output);
    rp = json_append(resp, rp, sizeof(resp), "},");
    rp = json_append_int(resp, rp, sizeof(resp), "tokens", n > 0 ? n : 0);
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_bool(resp, rp, sizeof(resp), "done", 1);

    /* Performance metrics */
    char num_buf[64];
    snprintf(num_buf, sizeof(num_buf), ",\"tokens_per_sec\":%.1f", tps);
    rp = json_append(resp, rp, sizeof(resp), num_buf);
    snprintf(num_buf, sizeof(num_buf), ",\"elapsed_ms\":%.0f", elapsed_ms);
    rp = json_append(resp, rp, sizeof(resp), num_buf);
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_int(resp, rp, sizeof(resp), "context_tokens", llm_chat_context_tokens());
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_int(resp, rp, sizeof(resp), "thinking_tokens", llm_thinking_tokens());

    /* Engine-side prefill + VRAM stats */
    {
        char mbuf[64];
        float pf = llm_last_prefill_ms();
        if (pf > 0) {
            snprintf(mbuf, sizeof(mbuf), ",\"prefill_ms\":%.0f", pf);
            rp = json_append(resp, rp, sizeof(resp), mbuf);
        }
        int vram = llm_last_vram_usage_mb();
        if (vram > 0) {
            rp = json_append(resp, rp, sizeof(resp), ",");
            rp = json_append_int(resp, rp, sizeof(resp), "vram_mb", vram);
        }
    }
    rp = json_append(resp, rp, sizeof(resp), "}");
    resp[rp] = '\0';

    send_json(sock, 200, resp);
}

/* ─── Request router ─── */
static void handle_request(socket_t client) {
    http_request_t req;
    memset(&req, 0, sizeof(req));

    if (parse_request(client, &req) < 0) {
        send_json(client, 400, "{\"error\":\"malformed request\"}");
        return;
    }

    /* CORS preflight */
    if (strcmp(req.method, "OPTIONS") == 0) {
        send_response(client, 200, "text/plain", "", 0);
        return;
    }

    /* Route — Geodessical native API */
    if (strcmp(req.path, "/") == 0 || strcmp(req.path, "/ui") == 0) {
        /* Serve the web chat UI */
        send_response(client, 200, "text/html", WEB_UI_HTML, WEB_UI_HTML_LEN);
    } else if (strcmp(req.path, "/health") == 0) {
        handle_health(client);
    } else if (strcmp(req.path, "/v1/version") == 0) {
        handle_version(client);
    } else if (strcmp(req.path, "/v1/models") == 0) {
        handle_tags(client);
    } else if (strcmp(req.path, "/v1/generate") == 0) {
        if (strcmp(req.method, "POST") != 0) {
            send_json(client, 405, "{\"error\":\"POST required\"}");
        } else {
            handle_generate(client, &req);
        }
    } else if (strcmp(req.path, "/v1/logprobs") == 0) {
        if (strcmp(req.method, "POST") != 0) {
            send_json(client, 405, "{\"error\":\"POST required\"}");
        } else {
            handle_logprobs(client, &req);
        }
    } else if (strcmp(req.path, "/v1/chat") == 0 ||
               strcmp(req.path, "/v1/chat/completions") == 0) {
        if (strcmp(req.method, "DELETE") == 0) {
            llm_reset_cache();
            send_json(client, 200, "{\"status\":\"chat reset\"}");
        } else if (strcmp(req.method, "POST") != 0) {
            send_json(client, 405, "{\"error\":\"POST or DELETE required\"}");
        } else {
            handle_chat(client, &req);
        }
    } else if (strcmp(req.path, "/mcp") == 0) {
        mcp_handle_request(client, req.method, req.body, req.body_len);
    } else {
        send_json(client, 404, "{\"error\":\"not found\"}");
    }
}

/* ─── Server main loop ─── */
int GD_api_serve(int port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        kprintf("[API] WSAStartup failed\n");
        return -1;
    }
#endif

    /* Initialise continuous-batching infrastructure */
    mutex_init(&g_infer_lock);
    condvar_init(&g_queue_ready);
    condvar_init(&g_result_ready);
    g_queue_head = g_queue_tail = g_queue_count = 0;

    /* Launch inference worker thread */
#ifdef _WIN32
    HANDLE worker = (HANDLE)_beginthreadex(NULL, 0, infer_worker, NULL, 0, NULL);
    if (!worker) {
        kprintf("[API] Failed to create inference worker thread\n");
        return -1;
    }
#else
    pthread_t worker;
    pthread_create(&worker, NULL, infer_worker, NULL);
#endif

    g_listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_sock == SOCKET_INVALID) {
        kprintf("[API] socket() failed\n");
        return -1;
    }

    /* Allow port reuse */
    int opt = 1;
    setsockopt(g_listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(g_listen_sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        kprintf("[API] bind() failed on port %d\n", port);
        sock_close(g_listen_sock);
        return -1;
    }

    if (listen(g_listen_sock, 8) != 0) {
        kprintf("[API] listen() failed\n");
        sock_close(g_listen_sock);
        return -1;
    }

    kprintf("[API] Geodessical serving on http://0.0.0.0:%d\n", port);
    kprintf("[API] Endpoints: /v1/generate, /v1/chat, /v1/models, /v1/version, /mcp\n");
    kprintf("[API] Continuous batching: enabled (queue depth %d)\n", CBATCH_QUEUE_MAX);
    kprintf("[API] Ready for inference\n\n");

    g_running = 1;
    while (g_running) {
        struct sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        socket_t client = accept(g_listen_sock, (struct sockaddr *)&client_addr,
                                  &client_len);
        if (client == SOCKET_INVALID) {
            if (!g_running) break; /* shutdown */
            continue;
        }

        /* Handle request: parse + enqueue + wait (inline, no extra thread needed
         * because the queue+condvar already decouples TCP I/O from inference) */
        handle_request(client);
        sock_close(client);
    }

    /* Signal worker to stop */
    mutex_lock(&g_infer_lock);
    g_running = 0;
    condvar_signal(&g_queue_ready);
    mutex_unlock(&g_infer_lock);

    sock_close(g_listen_sock);
    g_listen_sock = SOCKET_INVALID;

#ifdef _WIN32
    WSACleanup();
#endif

    kprintf("[API] Server stopped\n");
    return 0;
}

void GD_api_stop(void) {
    g_running = 0;
    if (g_listen_sock != SOCKET_INVALID) {
        sock_close(g_listen_sock);
    }
}
