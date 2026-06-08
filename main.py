


from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tunisian-low")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    





def get_responsefrompinecone(userquestion):
    vector = embedding_model.encode(userquestion).tolist()
    result = index.query(vector=vector, top_k=2, include_metadata=True)
    context = "\n".join([r["metadata"]["text"] for r in result["matches"]])
    return context








def get_response(userquestion,context):
    try:
    

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                                {"role": "system", "content": """You are an expert in Tunisian law and the Tunisian Constitution.
                You answer questions based ONLY on the provided context from official Tunisian legislation.

                Rules:
                - Only answer based on the context provided
                - If the answer is not in the context, say "I don't have enough information"
                - Always mention which article or chapter your answer comes from
                - Be precise and professional
                - You can answer in Arabic, French, or English — match the user's language
                """},
                {"role": "user", "content": f"""
Context from the Tunisian Constitution:
{context}

Question: {userquestion}
"""},
            ],
        )
        print(response.choices[0].message.content)

        
    except Exception as e:  
        print(f"An error occurred: {e}")



print("give me you question")
userquestion=input()

context = get_responsefrompinecone(userquestion)
get_response(userquestion, context)
