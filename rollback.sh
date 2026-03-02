#!/usr/bin/env bash
# AI Shaman (Ollama GPU Guardian) — Emergency Rollback
# Reverts Ollama to original ports and disables the guardian.
set -euo pipefail

echo "=== AI Shaman Rollback ==="
echo "Stopping and disabling guardian service..."
sudo systemctl stop ollama-guardian.service 2>/dev/null || true
sudo systemctl disable ollama-guardian.service 2>/dev/null || true

echo "Reverting Ollama GPU 0 to port 11434..."
sudo sed -i 's/127\.0\.0\.1:11444/0.0.0.0:11434/g' /etc/systemd/system/ollama.service.d/10-gpu-pin.conf

echo "Reverting Ollama GPU 1 to port 11435..."
sudo sed -i 's/127\.0\.0\.1:11445/0.0.0.0:11435/g' /etc/systemd/system/ollama-gpu1.service

echo "Removing GPU 1 guardian drop-in override..."
sudo rm -f /etc/systemd/system/ollama-gpu1.service.d/10-guardian-port.conf

echo "Reverting preload script ports..."
sed -i 's/localhost:11444/localhost:11434/g; s/localhost:11445/localhost:11435/g' /home/om/ollama-preload.sh

echo "Reloading systemd and restarting Ollama services..."
sudo systemctl daemon-reload
sudo systemctl restart ollama.service
sudo systemctl restart ollama-gpu1.service

echo "Waiting 5s for Ollama to start..."
sleep 5

echo "Verifying Ollama is responding on original ports..."
curl -sf http://localhost:11434/ && echo " <- GPU 0 OK" || echo " <- GPU 0 FAILED"
curl -sf http://localhost:11435/ && echo " <- GPU 1 OK" || echo " <- GPU 1 FAILED"

echo ""
echo "=== Rollback complete. Guardian disabled, Ollama on original ports. ==="
