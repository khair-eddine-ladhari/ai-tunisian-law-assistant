---
title: Tunisian Constitution Assistant
emoji: ⚖️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.17.3
app_file: main.py
pinned: false
---

# Tunisian Constitution Assistant

A RAG-based chatbot that answers questions about the Tunisian Constitution using Pinecone for retrieval and Groq (Llama 3.3 70B) for generation.

## Required Secrets

Set these in **Space Settings → Variables and secrets**:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`

Do **not** set `TRANSFORMERS_OFFLINE` (or set it to `0`) — the Space needs to download the embedding model on first run.

## Notes

The Pinecone index `tunisian-low` must already be populated with the constitution's embedded chunks (run your ingestion script locally before deploying).
