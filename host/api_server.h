/*
 * HyperTensor HTTP API Server
 *
 * Native HyperTensor REST API for model inference.
 *
 * Endpoints:
 *   POST /v1/generate    - Generate completion from prompt
 *   POST /v1/chat        - Multi-turn chat (with message history)
 *   GET  /v1/models      - List available models
 *   GET  /v1/version     - Server version info
 *   GET  /               - Health check
 *
 * Usage: hypertensor <model.gguf> --serve [--port 8080]
 *
 * Protocol: HTTP/1.1 synchronous JSON responses
 */

#ifndef HT_API_SERVER_H
#define HT_API_SERVER_H

/* Start the HTTP API server on the given port.
 * Blocks until server shutdown. Returns 0 on clean exit. */
int ht_api_serve(int port);

/* Stop the server gracefully. */
void ht_api_stop(void);

#endif /* HT_API_SERVER_H */
