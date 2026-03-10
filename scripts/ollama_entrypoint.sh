#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-llama3.2}"

ollama serve &
pid=$!

echo "Waiting for Ollama server to start..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Pulling model: $MODEL"
ollama pull "$MODEL"
echo "Model $MODEL is ready."

wait $pid
