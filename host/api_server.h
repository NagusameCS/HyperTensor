/*
 * HyperTensor HTTP API Server
 *
 * Provides an Ollama-compatible REST API for model inference,
 * enabling integration with Open WebUI, LangChain, and other tools.
 *
 * Endpoints:
 *   POST /api/generate    - Generate completion from prompt
 *   POST /api/chat        - Multi-turn chat (with message history)
 *   GET  /api/tags        - List available models
 *   GET  /api/version     - Server version info
 *   GET  /                - Health check
 *
 * Usage: hypertensor <model.gguf> --serve [--port 11434]
 *
 * Protocol: HTTP/1.1 with streaming NDJSON responses
 */

#ifndef HT_API_SERVER_H
#define HT_API_SERVER_H

/* Start the HTTP API server on the given port.
 * Blocks until server shutdown. Returns 0 on clean exit. */
int ht_api_serve(int port);

/* Stop the server gracefully. */
void ht_api_stop(void);

#endif /* HT_API_SERVER_H */
