import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "0")

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken
import gradio as gr

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tunisian-low")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# tiktoken — free, no gating, close enough for estimating any model's tokens
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

SYSTEM_PROMPT = """You are a Tunisian constitutional law expert.

Rules:
- Start your answer with the exact Article number (e.g. "According to Article 15...")
- Quote the relevant sentence directly from the context
- Then explain it in simple terms
- If the user talks with you generally like greeting, answer also with a greeting and say what you can help them with today in Tunisian constitutional law
- If the answer is NOT in the context, say ONLY: "This is not covered in the provided articles"
- Never answer from your own knowledge, only from the context
- If multiple articles are relevant, mention ALL of them
- If a question has multiple scenarios, explain each one separately
"""


def get_responsefrompinecone(userquestion):
    vector = embedding_model.encode(userquestion).tolist()
    result = index.query(vector=vector, top_k=5, include_metadata=True)

    matches = [r for r in result["matches"] if r["score"] > 0.45]
    if len(matches) == 0:
        matches = result["matches"][:2]

    context = "\n\n---\n\n".join([r["metadata"]["text"] for r in matches])
    return context


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(userquestion, context):
    full_prompt = f"""
Context from the Tunisian Constitution:
{context}

Question: {userquestion}
"""
    num_tokens = len(encoding.encode(full_prompt))

    if num_tokens > 6000:
        return "Message exceeds token limit, please ask a shorter question"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ],
    )
    return response.choices[0].message.content


def respond(message, history):
    try:
        context = get_responsefrompinecone(message)
        answer = get_response(message, context)
        return answer
    except Exception as e:
        return f"An error occurred: {e}"


demo = gr.ChatInterface(
    fn=respond,
    title="Tunisian Constitution Assistant",
    description=(
        "Ask questions about the Tunisian Constitution. "
        "Answers are based ONLY on the official constitutional text and include article citations."
    ),
    examples=[
        "What is the minimum age to become president?",
        "What are the rights of citizens?",
        "How is the constitution amended?",
    ],
)

if __name__ == "__main__":
    demo.launch()