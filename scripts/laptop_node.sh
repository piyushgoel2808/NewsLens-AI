#!/usr/bin/env bash
# Usage: ./scripts/laptop_node.sh [on|off|status]
set -e

ACTION="${1:-status}"
RENDER_URL="${RENDER_BACKEND_URL:-http://localhost:8000}"
NGROK_DOMAIN="${NGROK_STATIC_DOMAIN:-}"
ADMIN_PIN="${LAPTOP_ADMIN_PIN:-1234}"

if [ "$ACTION" = "on" ]; then
  echo "1. Ensuring local Ollama is running and EmbeddingGemma 2 is present..."
  pgrep -x "ollama" >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &)
  sleep 2

  echo "2. Starting HTTPS Tunnel to localhost:11434..."
  pkill -f "ngrok http 11434" 2>/dev/null || true
  if [ -n "$NGROK_DOMAIN" ]; then
    ngrok http 11434 --domain="$NGROK_DOMAIN" --log=stdout >/tmp/ngrok.log 2>&1 &
    TUNNEL_URL="https://$NGROK_DOMAIN"
  else
    ngrok http 11434 --log=stdout >/tmp/ngrok.log 2>&1 &
    sleep 3
    TUNNEL_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['tunnels'][0]['public_url'] if data.get('tunnels') else 'http://localhost:11434')" 2>/dev/null || echo "http://localhost:11434")
  fi

  echo "3. Enabling Sovereign Laptop Node on Backend ($RENDER_URL) -> $TUNNEL_URL ..."
  curl -s -X POST "$RENDER_URL/api/settings/laptop-node/toggle" \
    -H "Content-Type: application/json" \
    -H "X-Admin-Pin: $ADMIN_PIN" \
    -d "{\"enabled\": true, \"tunnel_url\": \"$TUNNEL_URL\"}" | python3 -m json.tool

  echo "✅ Laptop Local LLMs are now LIVE on your website!"

elif [ "$ACTION" = "off" ]; then
  echo "🔴 Turning OFF Sovereign Laptop Node on Backend ($RENDER_URL)..."
  curl -s -X POST "$RENDER_URL/api/settings/laptop-node/toggle" \
    -H "Content-Type: application/json" \
    -H "X-Admin-Pin: $ADMIN_PIN" \
    -d "{\"enabled\": false}" | python3 -m json.tool

  pkill -f "ngrok http 11434" 2>/dev/null || true
  echo "🛑 Tunnel stopped. Website users can no longer access your laptop."

else
  curl -s "$RENDER_URL/api/settings/laptop-node/status" | python3 -m json.tool
fi
