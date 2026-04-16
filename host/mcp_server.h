/*
 * Geodessical MCP (Model Context Protocol) Server
 *
 * Implements MCP 2024-11-05 over Streamable HTTP transport.
 * JSON-RPC 2.0 protocol with built-in tools for LLM inference.
 *
 * Built-in tools:
 *   generate      – Single-shot text completion
 *   chat          – Multi-turn chat with KV cache
 *   tokenize      – Text → token IDs
 *   model_info    – Model metadata
 *   reset_context – Clear chat KV cache
 *
 * Endpoints (wired into api_server.c):
 *   POST /mcp     – JSON-RPC message endpoint (+ SSE for streaming)
 *   GET  /mcp     – SSE stream for server-initiated notifications
 *   DELETE /mcp   – Session termination
 *
 * Reference: https://modelcontextprotocol.io/specification/2024-11-05
 */

#ifndef GD_MCP_SERVER_H
#define GD_MCP_SERVER_H

#ifdef _WIN32
#  include <winsock2.h>
typedef SOCKET mcp_socket_t;
#else
typedef int mcp_socket_t;
#endif

/* Handle an MCP request arriving on the given socket.
 * method: "GET", "POST", or "DELETE"
 * body: request body (JSON-RPC), may be NULL for GET/DELETE
 * body_len: length of body
 * Returns 0 on success, -1 on error. */
int mcp_handle_request(mcp_socket_t sock, const char *method,
                       const char *body, int body_len);

#endif /* GD_MCP_SERVER_H */
