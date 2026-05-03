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
 * Geodessical MCP Server — Implementation
 *
 * MCP 2024-11-05 over Streamable HTTP transport.
 * Self-contained JSON-RPC 2.0 handler with built-in inference tools.
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "mcp_server.h"
#include "hal.h"
#include "../runtime/nn/llm.h"

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  define sock_send(s, b, n) send((s), (b), (n), 0)
#else
#  include <sys/socket.h>
#  define sock_send(s, b, n) send((s), (b), (n), 0)
#endif

/* 
 * Constants
 *  */

#define MCP_PROTOCOL_VERSION  "2024-11-05"
#define MCP_SERVER_NAME       "Geodessical"
#define MCP_SERVER_VERSION    "0.5.0"
#define MCP_MAX_RESP          (512 * 1024)

/* 
 * JSON helpers (minimal, reuse patterns from api_server.c)
 *  */

static int ja(char *buf, int pos, int max, const char *s) {
    while (*s && pos < max - 1) buf[pos++] = *s++;
    return pos;
}

static int ja_str_val(char *buf, int pos, int max, const char *val) {
    pos = ja(buf, pos, max, "\"");
    while (*val && pos < max - 3) {
        if (*val == '"')       { buf[pos++] = '\\'; buf[pos++] = '"'; }
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

static int ja_kv(char *buf, int pos, int max, const char *key, const char *val) {
    pos = ja(buf, pos, max, "\"");
    pos = ja(buf, pos, max, key);
    pos = ja(buf, pos, max, "\":");
    pos = ja_str_val(buf, pos, max, val);
    return pos;
}

static int ja_ki(char *buf, int pos, int max, const char *key, int val) {
    pos = ja(buf, pos, max, "\"");
    pos = ja(buf, pos, max, key);
    pos = ja(buf, pos, max, "\":");
    char num[32];
    snprintf(num, sizeof(num), "%d", val);
    pos = ja(buf, pos, max, num);
    return pos;
}

static int ja_kb(char *buf, int pos, int max, const char *key, int val) {
    pos = ja(buf, pos, max, "\"");
    pos = ja(buf, pos, max, key);
    pos = ja(buf, pos, max, "\":");
    pos = ja(buf, pos, max, val ? "true" : "false");
    return pos;
}

/*  Mini JSON parser  */
static int jp_str(const char *json, const char *key, char *out, int max) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max - 1) {
        if (*p == '\\' && *(p + 1)) {
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

static int jp_int(const char *json, const char *key, int def) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return atoi(p);
}

static float jp_float(const char *json, const char *key, float def) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    return (float)atof(p);
}

/* Extract JSON-RPC id — could be int, string, or null */
typedef struct {
    int  is_string;
    int  int_val;
    char str_val[128];
    int  present;
} jsonrpc_id_t;

static void jp_id(const char *json, jsonrpc_id_t *id) {
    id->present = 0;
    id->is_string = 0;
    id->int_val = 0;
    id->str_val[0] = '\0';

    const char *p = strstr(json, "\"id\"");
    if (!p) return;
    p += 4;
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') {
        id->is_string = 1;
        id->present = 1;
        p++;
        int i = 0;
        while (*p && *p != '"' && i < 127) id->str_val[i++] = *p++;
        id->str_val[i] = '\0';
    } else if (*p == 'n') {
        /* null */
        id->present = 1;
    } else if (*p >= '0' && *p <= '9') {
        id->present = 1;
        id->int_val = atoi(p);
    } else if (*p == '-') {
        id->present = 1;
        id->int_val = atoi(p);
    }
}

/* Write JSON-RPC id value into buffer */
static int ja_id(char *buf, int pos, int max, const jsonrpc_id_t *id) {
    if (!id->present) {
        pos = ja(buf, pos, max, "null");
    } else if (id->is_string) {
        pos = ja_str_val(buf, pos, max, id->str_val);
    } else {
        char num[32];
        snprintf(num, sizeof(num), "%d", id->int_val);
        pos = ja(buf, pos, max, num);
    }
    return pos;
}

/* Find a sub-object by key, return pointer to the '{' */
static const char *jp_obj(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '{') return p;
    return NULL;
}

/* 
 * HTTP / SSE response helpers
 *  */

static void send_http(mcp_socket_t sock, int status, const char *content_type,
                      const char *body, int body_len) {
    const char *sstr = "200 OK";
    if (status == 202) sstr = "202 Accepted";
    else if (status == 400) sstr = "400 Bad Request";
    else if (status == 404) sstr = "404 Not Found";
    else if (status == 405) sstr = "405 Method Not Allowed";

    char hdr[512];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        sstr, content_type, body_len);
    sock_send(sock, hdr, hlen);
    if (body && body_len > 0)
        sock_send(sock, body, body_len);
}

static void send_jsonrpc(mcp_socket_t sock, const char *json) {
    send_http(sock, 200, "application/json", json, (int)strlen(json));
}

/* 
 * JSON-RPC error/result builders
 *  */

static void send_rpc_error(mcp_socket_t sock, const jsonrpc_id_t *id,
                           int code, const char *message) {
    char resp[1024];
    int p = 0;
    p = ja(resp, p, sizeof(resp), "{\"jsonrpc\":\"2.0\",\"id\":");
    p = ja_id(resp, p, sizeof(resp), id);
    p = ja(resp, p, sizeof(resp), ",\"error\":{");
    p = ja_ki(resp, p, sizeof(resp), "code", code);
    p = ja(resp, p, sizeof(resp), ",");
    p = ja_kv(resp, p, sizeof(resp), "message", message);
    p = ja(resp, p, sizeof(resp), "}}");
    resp[p] = '\0';
    send_jsonrpc(sock, resp);
}

/* 
 * Tool definitions (schema)
 *  */

static const char *TOOLS_LIST_JSON =
    "["
    "{"
        "\"name\":\"generate\","
        "\"description\":\"Generate a text completion from a prompt.\","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{"
                "\"prompt\":{\"type\":\"string\",\"description\":\"The input prompt\"},"
                "\"max_tokens\":{\"type\":\"integer\",\"description\":\"Maximum tokens to generate (default: 128)\"},"
                "\"temperature\":{\"type\":\"number\",\"description\":\"Sampling temperature (default: 0.7)\"}"
            "},"
            "\"required\":[\"prompt\"]"
        "}"
    "},"
    "{"
        "\"name\":\"chat\","
        "\"description\":\"Send a message in a multi-turn chat conversation. Maintains KV cache context between calls.\","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{"
                "\"message\":{\"type\":\"string\",\"description\":\"The user message\"},"
                "\"max_tokens\":{\"type\":\"integer\",\"description\":\"Maximum tokens to generate (default: 128)\"},"
                "\"temperature\":{\"type\":\"number\",\"description\":\"Sampling temperature (default: 0.7)\"}"
            "},"
            "\"required\":[\"message\"]"
        "}"
    "},"
    "{"
        "\"name\":\"tokenize\","
        "\"description\":\"Convert text into token IDs using the model's tokenizer.\","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{"
                "\"text\":{\"type\":\"string\",\"description\":\"Text to tokenize\"}"
            "},"
            "\"required\":[\"text\"]"
        "}"
    "},"
    "{"
        "\"name\":\"model_info\","
        "\"description\":\"Get information about the currently loaded model.\","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{}"
        "}"
    "},"
    "{"
        "\"name\":\"reset_context\","
        "\"description\":\"Reset the chat conversation context and KV cache.\","
        "\"inputSchema\":{"
            "\"type\":\"object\","
            "\"properties\":{}"
        "}"
    "}"
    "]";

/* 
 * Tool execution
 *  */

/* Execute "generate" tool */
static int tool_generate(const char *args_json, char *result, int max) {
    char prompt[8192];
    if (jp_str(args_json, "prompt", prompt, sizeof(prompt)) < 0)
        return snprintf(result, max, "[{\"type\":\"text\",\"text\":\"Error: missing 'prompt' argument\"}]");

    int max_tokens = jp_int(args_json, "max_tokens", 128);
    if (max_tokens > 4096) max_tokens = 4096;
    if (max_tokens < 1) max_tokens = 1;

    static char output[131072];
    output[0] = '\0';
    int n = llm_prompt_n(prompt, output, (int)sizeof(output), max_tokens);

    int p = 0;
    p = ja(result, p, max, "[{\"type\":\"text\",\"text\":");
    p = ja_str_val(result, p, max, output);
    p = ja(result, p, max, "}]");
    result[p] = '\0';
    (void)n;
    return p;
}

/* Execute "chat" tool */
static int tool_chat(const char *args_json, char *result, int max) {
    char message[8192];
    if (jp_str(args_json, "message", message, sizeof(message)) < 0)
        return snprintf(result, max, "[{\"type\":\"text\",\"text\":\"Error: missing 'message' argument\"}]");

    int max_tokens = jp_int(args_json, "max_tokens", 128);
    if (max_tokens > 4096) max_tokens = 4096;
    if (max_tokens < 1) max_tokens = 1;
    float temp = jp_float(args_json, "temperature", 0.7f);

    static char output[131072];
    output[0] = '\0';
    int n = llm_chat_turn(message, output, (int)sizeof(output), max_tokens, temp);

    int p = 0;
    p = ja(result, p, max, "[{\"type\":\"text\",\"text\":");
    p = ja_str_val(result, p, max, output);
    p = ja(result, p, max, "}]");
    result[p] = '\0';
    (void)n;
    return p;
}

/* Execute "tokenize" tool */
static int tool_tokenize(const char *args_json, char *result, int max) {
    char text[8192];
    if (jp_str(args_json, "text", text, sizeof(text)) < 0)
        return snprintf(result, max, "[{\"type\":\"text\",\"text\":\"Error: missing 'text' argument\"}]");

    int tokens[4096];
    int n = llm_tokenize_text(text, tokens, 4096);

    int p = 0;
    p = ja(result, p, max, "[{\"type\":\"text\",\"text\":\"[");
    for (int i = 0; i < n && p < max - 32; i++) {
        char num[16];
        snprintf(num, sizeof(num), "%s%d", i > 0 ? "," : "", tokens[i]);
        p = ja(result, p, max, num);
    }
    p = ja(result, p, max, "]\"}]");
    result[p] = '\0';
    return p;
}

/* Execute "model_info" tool */
static int tool_model_info(const char *args_json, char *result, int max) {
    (void)args_json;
    int p = 0;
    p = ja(result, p, max, "[{\"type\":\"text\",\"text\":\"{");

    /* Build escaped JSON string inside text content */
    char info[2048];
    int ip = 0;
    ip = ja_kv(info, ip, sizeof(info), "model", llm_model_name());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_kv(info, ip, sizeof(info), "arch", llm_model_arch());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_ki(info, ip, sizeof(info), "layers", llm_model_layers());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_ki(info, ip, sizeof(info), "dim", llm_model_dim());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_ki(info, ip, sizeof(info), "vocab", llm_model_vocab());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_kv(info, ip, sizeof(info), "backend", llm_backend_name());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_ki(info, ip, sizeof(info), "context_tokens", llm_chat_context_tokens());
    ip = ja(info, ip, sizeof(info), ",");
    ip = ja_ki(info, ip, sizeof(info), "context_max", llm_chat_context_max());
    int vram = llm_last_vram_usage_mb();
    if (vram > 0) {
        ip = ja(info, ip, sizeof(info), ",");
        ip = ja_ki(info, ip, sizeof(info), "vram_mb", vram);
    }
    info[ip] = '\0';

    /* Escape the inner JSON for the text field */
    p = ja(result, p, max, "[{\"type\":\"text\",\"text\":\"{");
    for (int i = 0; i < ip && p < max - 8; i++) {
        char c = info[i];
        if (c == '"') { result[p++] = '\\'; result[p++] = '"'; }
        else if (c == '\\') { result[p++] = '\\'; result[p++] = '\\'; }
        else result[p++] = c;
    }
    p = ja(result, p, max, "}\"}]");
    result[p] = '\0';
    return p;
}

/* Execute "reset_context" tool */
static int tool_reset_context(const char *args_json, char *result, int max) {
    (void)args_json;
    llm_reset_cache();
    return snprintf(result, max,
        "[{\"type\":\"text\",\"text\":\"Chat context and KV cache reset.\"}]");
}

/* Dispatch tool by name */
static int dispatch_tool(const char *name, const char *args_json,
                         char *result, int max) {
    if (strcmp(name, "generate") == 0)      return tool_generate(args_json, result, max);
    if (strcmp(name, "chat") == 0)          return tool_chat(args_json, result, max);
    if (strcmp(name, "tokenize") == 0)      return tool_tokenize(args_json, result, max);
    if (strcmp(name, "model_info") == 0)    return tool_model_info(args_json, result, max);
    if (strcmp(name, "reset_context") == 0) return tool_reset_context(args_json, result, max);
    return -1; /* unknown tool */
}

/* 
 * MCP method handlers
 *  */

static void handle_initialize(mcp_socket_t sock, const jsonrpc_id_t *id,
                              const char *params) {
    (void)params;
    char resp[2048];
    int p = 0;
    p = ja(resp, p, sizeof(resp), "{\"jsonrpc\":\"2.0\",\"id\":");
    p = ja_id(resp, p, sizeof(resp), id);
    p = ja(resp, p, sizeof(resp), ",\"result\":{");
    p = ja_kv(resp, p, sizeof(resp), "protocolVersion", MCP_PROTOCOL_VERSION);
    p = ja(resp, p, sizeof(resp), ",\"capabilities\":{");
    p = ja(resp, p, sizeof(resp), "\"tools\":{\"listChanged\":false}");
    p = ja(resp, p, sizeof(resp), "},\"serverInfo\":{");
    p = ja_kv(resp, p, sizeof(resp), "name", MCP_SERVER_NAME);
    p = ja(resp, p, sizeof(resp), ",");
    p = ja_kv(resp, p, sizeof(resp), "version", MCP_SERVER_VERSION);
    p = ja(resp, p, sizeof(resp), "}}}");
    resp[p] = '\0';
    send_jsonrpc(sock, resp);
}

static void handle_tools_list(mcp_socket_t sock, const jsonrpc_id_t *id) {
    char *resp = (char *)malloc(MCP_MAX_RESP);
    if (!resp) {
        send_rpc_error(sock, id, -32603, "internal error: alloc failed");
        return;
    }
    int p = 0;
    p = ja(resp, p, MCP_MAX_RESP, "{\"jsonrpc\":\"2.0\",\"id\":");
    p = ja_id(resp, p, MCP_MAX_RESP, id);
    p = ja(resp, p, MCP_MAX_RESP, ",\"result\":{\"tools\":");
    p = ja(resp, p, MCP_MAX_RESP, TOOLS_LIST_JSON);
    p = ja(resp, p, MCP_MAX_RESP, "}}");
    resp[p] = '\0';
    send_jsonrpc(sock, resp);
    free(resp);
}

static void handle_tools_call(mcp_socket_t sock, const jsonrpc_id_t *id,
                              const char *params) {
    char tool_name[128];
    if (jp_str(params, "name", tool_name, sizeof(tool_name)) < 0) {
        send_rpc_error(sock, id, -32602, "missing 'name' in params");
        return;
    }

    /* Find arguments sub-object */
    const char *args = jp_obj(params, "arguments");
    const char *args_str = args ? args : "{}";

    char *result = (char *)malloc(MCP_MAX_RESP);
    if (!result) {
        send_rpc_error(sock, id, -32603, "internal error: alloc failed");
        return;
    }

    int n = dispatch_tool(tool_name, args_str, result, MCP_MAX_RESP);
    if (n < 0) {
        char err[256];
        snprintf(err, sizeof(err), "unknown tool: %s", tool_name);
        send_rpc_error(sock, id, -32602, err);
        free(result);
        return;
    }

    /* Build successful tool result */
    char *resp = (char *)malloc(MCP_MAX_RESP);
    if (!resp) {
        send_rpc_error(sock, id, -32603, "internal error: alloc failed");
        free(result);
        return;
    }

    int p = 0;
    p = ja(resp, p, MCP_MAX_RESP, "{\"jsonrpc\":\"2.0\",\"id\":");
    p = ja_id(resp, p, MCP_MAX_RESP, id);
    p = ja(resp, p, MCP_MAX_RESP, ",\"result\":{\"content\":");
    /* result is already a JSON array of content blocks */
    p = ja(resp, p, MCP_MAX_RESP, result);
    p = ja(resp, p, MCP_MAX_RESP, ",");
    p = ja_kb(resp, p, MCP_MAX_RESP, "isError", 0);
    p = ja(resp, p, MCP_MAX_RESP, "}}");
    resp[p] = '\0';

    send_jsonrpc(sock, resp);
    free(resp);
    free(result);
}

static void handle_ping(mcp_socket_t sock, const jsonrpc_id_t *id) {
    char resp[256];
    int p = 0;
    p = ja(resp, p, sizeof(resp), "{\"jsonrpc\":\"2.0\",\"id\":");
    p = ja_id(resp, p, sizeof(resp), id);
    p = ja(resp, p, sizeof(resp), ",\"result\":{}}");
    resp[p] = '\0';
    send_jsonrpc(sock, resp);
}

/* 
 * SSE (Server-Sent Events) Stream Implementation
 *
 * GET /mcp establishes a persistent SSE connection for server-initiated
 * notifications. This enables the server to push events to clients in
 * real-time without polling.
 *
 * Event types:
 *   ping       – Keep-alive heartbeat (every 30s)
 *   progress   – Generation progress updates during tool execution
 *   notify     – Server-initiated notifications
 *  */

/* SSE connection state */
static volatile int g_sse_active = 0;
static mcp_socket_t g_sse_socket = -1;

/* Send an SSE event with the given type and data */
static int sse_send_event(mcp_socket_t sock, const char *event_type,
                          const char *data) {
    if (sock < 0) return -1;
    
    char buf[4096];
    int len = snprintf(buf, sizeof(buf),
                       "event: %s\n"
                       "data: %s\n\n",
                       event_type, data);
    if (len < 0 || len >= (int)sizeof(buf)) return -1;
    
    int sent = sock_send(sock, buf, len);
    return (sent == len) ? 0 : -1;
}

/* Send an SSE ping (keep-alive) */
static int sse_send_ping(mcp_socket_t sock) {
    char data[128];
    snprintf(data, sizeof(data),
             "{\"type\":\"ping\",\"timestamp\":%lu}",
             (unsigned long)time(NULL));
    return sse_send_event(sock, "ping", data);
}

/* Send a progress update */
int mcp_sse_progress(int tokens_done, int tokens_total, const char *partial_text) {
    if (!g_sse_active || g_sse_socket < 0) return -1;
    
    char data[2048];
    int p = 0;
    p = ja(data, p, sizeof(data), "{\"type\":\"progress\",");
    p = ja_ki(data, p, sizeof(data), "tokensDone", tokens_done);
    p = ja(data, p, sizeof(data), ",");
    p = ja_ki(data, p, sizeof(data), "tokensTotal", tokens_total);
    if (partial_text && *partial_text) {
        p = ja(data, p, sizeof(data), ",");
        p = ja_kv(data, p, sizeof(data), "text", partial_text);
    }
    p = ja(data, p, sizeof(data), "}");
    data[p] = '\0';
    
    return sse_send_event(g_sse_socket, "progress", data);
}

/* Send a server notification */
int mcp_sse_notify(const char *method, const char *params_json) {
    if (!g_sse_active || g_sse_socket < 0) return -1;
    
    char data[2048];
    int p = 0;
    p = ja(data, p, sizeof(data), "{\"jsonrpc\":\"2.0\",\"method\":");
    p = ja_str_val(data, p, sizeof(data), method);
    if (params_json && *params_json) {
        p = ja(data, p, sizeof(data), ",\"params\":");
        p = ja(data, p, sizeof(data), params_json);
    }
    p = ja(data, p, sizeof(data), "}");
    data[p] = '\0';
    
    return sse_send_event(g_sse_socket, "notify", data);
}

/* Close the SSE connection */
static void sse_close(void) {
    g_sse_active = 0;
    g_sse_socket = -1;
}

/* Handle GET /mcp - establish SSE connection */
static int handle_sse_stream(mcp_socket_t sock) {
    /* Send SSE headers */
    const char *headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    
    int hlen = (int)strlen(headers);
    if (sock_send(sock, headers, hlen) != hlen) {
        return -1;
    }
    
    /* Register as active SSE connection */
    g_sse_socket = sock;
    g_sse_active = 1;
    
    /* Send initial connection event */
    char init_data[256];
    snprintf(init_data, sizeof(init_data),
             "{\"type\":\"connected\",\"server\":\"%s\",\"version\":\"%s\"}",
             MCP_SERVER_NAME, MCP_SERVER_VERSION);
    sse_send_event(sock, "init", init_data);
    
    /* Send initial ping */
    sse_send_ping(sock);
    
    return 0;  /* Connection stays open for future events */
}

/* 
 * Main MCP request dispatcher
 *  */

int mcp_handle_request(mcp_socket_t sock, const char *method,
                       const char *body, int body_len) {

    /* GET /mcp — SSE stream endpoint for server-initiated notifications */
    if (strcmp(method, "GET") == 0) {
        return handle_sse_stream(sock);
    }

    /* DELETE /mcp — session termination */
    if (strcmp(method, "DELETE") == 0) {
        sse_close();  /* Close any active SSE connection */
        llm_reset_cache();
        send_http(sock, 200, "application/json",
                  "{\"status\":\"session terminated\"}", 33);
        return 0;
    }

    /* POST /mcp — JSON-RPC message */
    if (strcmp(method, "POST") != 0) {
        send_http(sock, 405, "application/json",
                  "{\"error\":\"method not allowed\"}", 30);
        return -1;
    }

    if (!body || body_len == 0) {
        jsonrpc_id_t null_id = {0};
        send_rpc_error(sock, &null_id, -32700, "empty request body");
        return -1;
    }

    /* Parse JSON-RPC envelope */
    char rpc_method[128];
    if (jp_str(body, "method", rpc_method, sizeof(rpc_method)) < 0) {
        jsonrpc_id_t null_id = {0};
        send_rpc_error(sock, &null_id, -32600, "missing 'method' field");
        return -1;
    }

    jsonrpc_id_t id;
    jp_id(body, &id);

    /* Find params */
    const char *params = jp_obj(body, "params");
    const char *params_str = params ? params : body; /* fallback to full body */

    /* Dispatch by MCP method */
    if (strcmp(rpc_method, "initialize") == 0) {
        handle_initialize(sock, &id, params_str);
    } else if (strcmp(rpc_method, "ping") == 0) {
        handle_ping(sock, &id);
    } else if (strcmp(rpc_method, "notifications/initialized") == 0) {
        /* Client acknowledgement — no response needed for notifications */
        send_http(sock, 202, "application/json", "{}", 2);
    } else if (strcmp(rpc_method, "tools/list") == 0) {
        handle_tools_list(sock, &id);
    } else if (strcmp(rpc_method, "tools/call") == 0) {
        handle_tools_call(sock, &id, params_str);
    } else {
        send_rpc_error(sock, &id, -32601, "method not found");
    }

    return 0;
}
