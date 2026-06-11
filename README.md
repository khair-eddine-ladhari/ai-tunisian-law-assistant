# 🇹🇳 Tunisian Constitution Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the **Tunisian Constitution** with precise, cited responses. Built with semantic search over the official constitutional text, the assistant grounds every answer in the exact article it comes from — reducing hallucination and giving users verifiable, trustworthy information about their legal rights and government structure.

🔗 **Live demo:** [Hugging Face Space](https://huggingface.co/spaces/khaireddineladhari/tunisian-constitution-assistant)

---

## ✨ Features

- **Article-accurate citations** — every answer references the specific constitutional article it's based on (e.g. *"According to Article 40..."*)
- **Strict grounding** — the model answers *only* from retrieved context and explicitly says when information isn't covered, instead of hallucinating
- **Multilingual** — understands and responds in Arabic, French, and English
- **Multi-article reasoning** — combines and explains multiple relevant articles when a question touches several provisions
- **Semantic search** — retrieves relevant articles by meaning, not just keyword matching
- **Resilient** — automatic retries with exponential backoff for transient API/network failures
- **Token-safe** — pre-checks prompt size to avoid oversized requests
- **Clean chat UI** — ChatGPT/Claude-style interface built with Gradio

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF of     │────▶│  Chunking by      │────▶│  Embedding with  │
│ Constitution │     │  Article number   │     │ all-MiniLM-L6-v2 │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  Pinecone Vector  │
                                                 │      Index        │
                                                 └────────┬─────────┘
                                                          │
User question ──▶ Embed question ──▶ Semantic search ───┘
                                              │
                                              ▼
                                  Top-k articles (score > 0.45)
                                              │
                                              ▼
                              ┌───────────────────────────────┐
                              │  Llama 3.3 70B (via Groq API)   │
                              │  + strict system prompt         │
                              └───────────────┬───────────────┘
                                              │
                                              ▼
                                 Cited, grounded answer
                                  (Gradio chat interface)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Embeddings** | [`sentence-transformers`](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| **Vector database** | [Pinecone](https://www.pinecone.io/) (serverless) |
| **LLM inference** | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| **PDF parsing** | [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) |
| **Token counting** | [`tiktoken`](https://github.com/openai/tiktoken) |
| **Retry/resilience** | [`tenacity`](https://github.com/jd/tenacity) |
| **UI** | [Gradio](https://www.gradio.app/) |
| **Hosting** | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## 📂 Project Structure

```
.
├── ingestion.py        # One-time script: PDF → chunks → embeddings → Pinecone
├── main.py              # RAG query pipeline + Gradio chat UI (deployed app)
├── requirements.txt      # Python dependencies
├── .env                 # API keys (not committed)
└── README.md
```

---

## ⚙️ How It Works

### 1. Ingestion (run once)
- The constitution PDF is parsed and split **by article** (`Article 1`, `Article 2`, ...) — preserving each article as a coherent semantic unit
- Articles longer than 1000 characters are split by sentence, with the article header re-attached to each piece
- Fragments shorter than 100 characters (headers, empty splits) are discarded
- Each chunk is embedded with `all-MiniLM-L6-v2` and upserted into a Pinecone index, with the article number extracted via regex and stored in metadata

### 2. Query pipeline
- The user's question is embedded and matched against Pinecone (`top_k=5`)
- Results scoring below `0.45` are filtered out, with a fallback to the top 2 matches if nothing passes the threshold
- The retrieved article texts are assembled into context and sent to **Llama 3.3 70B** via Groq, alongside a system prompt that enforces:
  - Citing the exact article number
  - Quoting directly from the context
  - Explaining the article in plain language
  - Refusing to answer if the information isn't present
  - Covering **all** relevant articles for multi-faceted questions
- A `tiktoken`-based check prevents oversized prompts from being sent
- API calls use exponential-backoff retries to handle rate limits and transient errors gracefully

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- A [Pinecone API key](https://www.pinecone.io/) (free tier available)

### Installation

```bash
git clone https://github.com/<your-username>/tunisian-constitution-assistant.git
cd tunisian-constitution-assistant
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
TRANSFORMERS_OFFLINE=0
```

### 1. Build the index (one-time)

```bash
python ingestion.py
```

This parses the constitution PDF, chunks it by article, generates embeddings, and populates your Pinecone index (`tunisian-low`).

### 2. Run the app

```bash
python main.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`) to use the chat interface.

---

## ☁️ Deployment (Hugging Face Spaces)

1. Create a new **Gradio** Space on Hugging Face
2. Push `main.py`, `requirements.txt`, and `README.md` to the Space repo
3. Add the following **Secrets** in Space Settings:
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
4. The Space builds automatically and serves the Gradio UI at your Space's public URL

> Note: `TRANSFORMERS_OFFLINE` should be left unset (or `0`) in production so the embedding model can be downloaded on first run.

---

## 🧪 Example Questions

| Question | Behavior |
|---|---|
| *"What is the minimum age to become president?"* | Cites Article 40 with the exact age requirement |
| *"What percentage of votes is needed to amend the constitution?"* | Combines Articles 76 & 77, explaining both the referendum and non-referendum paths |
| *"What does the constitution say about the Mufti of the Republic?"* | Correctly responds: *"This is not covered in the provided articles"* |
| *"Hi"* | Responds with a friendly greeting and offers help on constitutional topics |

---

## 🔮 Possible Improvements

- [ ] Switch to a multilingual embedding model for stronger Arabic semantic search
- [ ] Add hybrid search (semantic + keyword reranking)
- [ ] Expand corpus to include related legislation (e.g. penal code) with clear source labeling
- [ ] Add conversation memory for follow-up questions
- [ ] Cache frequent queries to reduce API costs

---

## ⚠️ Disclaimer

This project is for **informational and educational purposes only**. It is not a substitute for professional legal advice. Always consult a qualified lawyer for legal matters.

---

## 📄 License

MIT License — feel free to use, modify, and build upon this project.
