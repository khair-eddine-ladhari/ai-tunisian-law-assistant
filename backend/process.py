

from concurrent.futures import process


import fitz
import json
import os








import fitz

pdf_path = "../data/pdfs/constitution_tunisie.pdf"

doc = fitz.open(pdf_path)

text = ""
for page in doc:
    text += page.get_text()

doc.close()

print(text[:500])  # print first 500 chars to test







from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))



pc.create_index(
    name="tunisian-low",
    dimension=384,
 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)


index=pc.Index("tunisian-low")





#chunk ones
articles = text.split("الفصل")

chunked_text = []
for article in articles:
    article = "الفصل " + article
    if len(article) > 1000:
        # split long articles into pieces of 1000 chars
        for i in range(0, len(article), 1000):
            chunked_text.append(article[i:i+1000])
    else:
        chunked_text.append(article)

        
 # split text into chunks of 1000 words
embeddings = model.encode(chunked_text)


vectors = []
for i, (chunk, embedding) in enumerate(zip(chunked_text, embeddings)):
    vector = {
        "id": f"chunk_{i}",
        "values": embedding.tolist(),
        "metadata": {"text": chunk},
    }
    vectors.append(vector)

index.upsert(vectors=vectors)