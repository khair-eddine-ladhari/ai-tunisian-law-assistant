import os
from dotenv import load_dotenv
load_dotenv()

# must be FIRST before any other import
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE")

# now import everything else
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tunisian-low")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")




def get_responsefrompinecone(userquestion):
    vector = embedding_model.encode(userquestion).tolist()
    result = index.query(vector=vector, top_k=5, include_metadata=True) 
    for r in result["matches"]:
        print(f"Score: {r['score']:.3f} | {r['metadata']['text'][:80]}") # was 2
    
    # Filter low-relevance results
    matches = [r for r in result["matches"] if r["score"] > 0.45]
    if len(matches) == 0:
        matches = result["matches"][:2]
    
    context = "\n\n---\n\n".join([r["metadata"]["text"] for r in matches])
    return context






def get_response(userquestion,context):
    try:
    

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                               {"role": "system", "content": """You are a Tunisian constitutional law expert.

Rules:
- Start your answer with the exact Article number (e.g. "According to Article 15...")
- Quote the relevant sentence directly from the context
- Then explain it in simple terms
-if the user talk with you generally like greating ; answere also with greating and say what i can help you today in the tunsian constutional law 
- If the answer is NOT in the context, say ONLY: "This is not covered in the provided articles"
- Never answer from your own knowledge, only from the context
- If multiple articles are relevant, mention ALL of them
- If a question has multiple scenarios, explain each one separately
"""},
                {"role": "user", "content": f"""
Question: {userquestion}

Relevant articles from the Tunisian Constitution:
{context}

Answer based strictly on the articles above.
"""},
            ],
            temperature=0.1
        )
        print(response.choices[0].message.content)

        
    except Exception as e:  
        print(f"An error occurred: {e}")




# at the bottom of main.py — replace this:
print("give me you question")
userquestion = input()
context = get_responsefrompinecone(userquestion)
get_response(userquestion, context)

# with this:
print("Model ready! Type 'quit' to exit.\n")
while True:
    print("Your question:")
    userquestion = input()
    
    if userquestion.lower() == "quit":
        break
    
    context = get_responsefrompinecone(userquestion)
    get_response(userquestion, context)
    print()  # empty line between answers