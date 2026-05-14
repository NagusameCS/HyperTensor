# HyperTensor deployment recipes

Production deployment templates for the `ht-repro` API server.

## Quick start (Docker Compose)

```bash
cd deploy
cp .env.example .env
# edit .env, set HT_REPRO_TOKEN to a random string
docker compose up -d
```

The server is then available at `http://localhost:8765` (UI) and
`http://localhost:8765/api/v1/` (REST API). With a token set, calls must include
`Authorization: Bearer $HT_REPRO_TOKEN`.

## Files

- `Dockerfile.ht-repro` — small image with `ht-repro` installed
- `docker-compose.yml`  — single-node service + persistent volume for `~/.ht-repro`
- `nginx.conf`          — reverse proxy with TLS termination + rate limiting
- `ht-repro.service`    — systemd unit for bare-metal Linux deployment
- `terraform/`          — AWS EC2 (L40S) recipe (minimal, BYO state backend)
- `.env.example`        — environment variables

## Bare-metal (systemd + nginx)

```bash
sudo cp ht-repro.service /etc/systemd/system/
sudo cp nginx.conf /etc/nginx/sites-available/ht-repro
sudo ln -s /etc/nginx/sites-available/ht-repro /etc/nginx/sites-enabled/
sudo systemctl daemon-reload
sudo systemctl enable --now ht-repro
sudo nginx -t && sudo systemctl reload nginx
```

## Cloud (AWS)

See [`terraform/`](terraform/). Provisions a single g6e.xlarge (L40S 48GB)
with the docker-compose stack pre-baked into user-data.

## Hardening checklist

- [ ] Set `HT_REPRO_TOKEN` to ≥32 random bytes
- [ ] Enable TLS in `nginx.conf` (uncomment the `ssl_*` lines)
- [ ] Restrict `/api/v1/infer` and `/api/v1/graft` to authorized IPs via the
      `allow`/`deny` directives in nginx
- [ ] Mount `~/.ht-repro` on encrypted storage
- [ ] Add `fail2ban` jail for 401s in nginx access log
