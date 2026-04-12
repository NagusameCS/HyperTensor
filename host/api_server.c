/*
 * HyperTensor HTTP API Server
 *
 * Native HyperTensor REST API for LLM inference.
 * Uses raw Winsock2 — no external dependencies.
 */

#define _CRT_SECURE_NO_WARNINGS
#include "api_server.h"
#include "hal.h"
#include "../runtime/nn/llm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32")
typedef SOCKET socket_t;
#define SOCKET_INVALID INVALID_SOCKET
#define sock_close closesocket
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
typedef int socket_t;
#define SOCKET_INVALID (-1)
#define sock_close close
#endif

/* ─── State ─── */
static volatile int g_running = 1;
static socket_t g_listen_sock = SOCKET_INVALID;

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
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
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
    send_json(sock, 200, "{\"name\":\"HyperTensor\",\"version\":\"0.5.0\",\"backend\":\"cuda\"}");
}

static void handle_health(socket_t sock) {
    send_json(sock, 200, "{\"status\":\"ok\"}");
}

static void handle_tags(socket_t sock) {
    char resp[1024];
    int p = 0;
    p = json_append(resp, p, sizeof(resp), "{\"models\":[{");
    p = json_append_str(resp, p, sizeof(resp), "name", llm_model_name());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_str(resp, p, sizeof(resp), "model", llm_model_name());
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_str(resp, p, sizeof(resp), "modified_at", "2025-01-01T00:00:00Z");
    p = json_append(resp, p, sizeof(resp), ",");
    p = json_append_int(resp, p, sizeof(resp), "size", 0);
    p = json_append(resp, p, sizeof(resp), "}]}");
    resp[p] = '\0';
    send_json(sock, 200, resp);
}

static void handle_generate(socket_t sock, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        send_json(sock, 400, "{\"error\":\"empty request body\"}");
        return;
    }

    /* Parse request JSON */
    char prompt[8192];
    if (json_find_str(req->body, "prompt", prompt, sizeof(prompt)) < 0) {
        send_json(sock, 400, "{\"error\":\"missing 'prompt' field\"}");
        return;
    }

    int max_tokens = json_find_int(req->body, "num_predict", 128);
    if (max_tokens > 4096) max_tokens = 4096;
    float temp = json_find_float(req->body, "temperature", 0.7f);
    (void)temp; /* TODO: pass to inference */

    /* Generate response */
    static char output[131072];
    output[0] = '\0';
    int n = llm_prompt_n(prompt, output, (int)sizeof(output), max_tokens);

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
    p = json_append(resp, p, sizeof(resp), "}");
    resp[p] = '\0';

    send_json(sock, 200, resp);
}

static void handle_chat(socket_t sock, http_request_t *req) {
    if (!req->body || req->body_len == 0) {
        send_json(sock, 400, "{\"error\":\"empty request body\"}");
        return;
    }

    /* Extract the last user message from the messages array.
     * Simple approach: find the last "content" value after the last "user" role. */
    char content[8192];
    content[0] = '\0';

    /* Find the last occurrence of "role":"user" */
    const char *last_user = (const char *)0;
    const char *p = req->body;
    while ((p = strstr(p, "\"user\"")) != (const char *)0) {
        last_user = p;
        p += 6;
    }

    if (last_user) {
        /* Find "content" after this user role */
        const char *c = strstr(last_user, "\"content\"");
        if (c) {
            /* Extract the content value */
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

    if (content[0] == '\0') {
        send_json(sock, 400, "{\"error\":\"no user message found\"}");
        return;
    }

    int max_tokens = json_find_int(req->body, "num_predict", 128);
    if (max_tokens > 4096) max_tokens = 4096;

    /* Use chat turn for multi-turn context */
    static char output[131072];
    output[0] = '\0';
    int n = llm_chat_turn(content, output, (int)sizeof(output), max_tokens, 0.7f);

    /* Build chat response */
    char resp[MAX_RESP_SIZE];
    int rp = 0;
    rp = json_append(resp, rp, sizeof(resp), "{");
    rp = json_append_str(resp, rp, sizeof(resp), "model", llm_model_name());
    rp = json_append(resp, rp, sizeof(resp), ",\"message\":{");
    rp = json_append_str(resp, rp, sizeof(resp), "role", "assistant");
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_str(resp, rp, sizeof(resp), "content", output);
    rp = json_append(resp, rp, sizeof(resp), "},");
    rp = json_append_bool(resp, rp, sizeof(resp), "done", 1);
    rp = json_append(resp, rp, sizeof(resp), ",");
    rp = json_append_int(resp, rp, sizeof(resp), "eval_count", n > 0 ? n : 0);
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

    /* Route — HyperTensor native API */
    if (strcmp(req.path, "/") == 0 || strcmp(req.path, "/health") == 0) {
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
    } else if (strcmp(req.path, "/v1/chat") == 0) {
        if (strcmp(req.method, "POST") != 0) {
            send_json(client, 405, "{\"error\":\"POST required\"}");
        } else {
            handle_chat(client, &req);
        }
    } else {
        send_json(client, 404, "{\"error\":\"not found\"}");
    }
}

/* ─── Server main loop ─── */
int ht_api_serve(int port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        kprintf("[API] WSAStartup failed\n");
        return -1;
    }
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

    kprintf("[API] HyperTensor serving on http://0.0.0.0:%d\n", port);
    kprintf("[API] Endpoints: /v1/generate, /v1/chat, /v1/models, /v1/version\n");
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

        /* Handle request synchronously (single-threaded for safety with
         * static inference buffers — matches Ollama's behavior) */
        handle_request(client);
        sock_close(client);
    }

    sock_close(g_listen_sock);
    g_listen_sock = SOCKET_INVALID;

#ifdef _WIN32
    WSACleanup();
#endif

    kprintf("[API] Server stopped\n");
    return 0;
}

void ht_api_stop(void) {
    g_running = 0;
    if (g_listen_sock != SOCKET_INVALID) {
        sock_close(g_listen_sock);
    }
}
