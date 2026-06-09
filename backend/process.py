

from concurrent.futures import process


import fitz
import json
import os
import re







import fitz

pdf_path = "./data/pdfs/Tunisiaconstitution.pdf"

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
articles = text.split("Article")

chunked_text = []
for i, article in enumerate(articles):
    if i == 0:
        continue  # skip garbage before first article
    
    article = ("Article " + article).strip()
    
    # Only split if the article is genuinely huge
    if len(article) > 1000:
        # Split by sentence instead of hard character cut
        sentences = article.split(".")
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) < 1000:
                current += sentence + "."
            else:
                chunked_text.append(current.strip())
                current = "Article " + sentence + "."  # keep article header
        if current:
            chunked_text.append(current.strip())
    else:
        chunked_text.append(article)
 # split text into chunks of 1000 words
chunked_text = [c for c in chunked_text if len(c.strip()) > 100]
embeddings = model.encode(chunked_text)



vectors = []
for i, (chunk, embedding) in enumerate(zip(chunked_text, embeddings)):
    
    # extract real article number from chunk text
    match = re.search(r'Article\s+(\d+)', chunk)
    article_num = match.group(1) if match else str(i)
    
    vector = {
        "id": f"chunk_{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": chunk,
            "article_number": article_num,  # "7" instead of loop index
        },
    }
    vectors.append(vector)

index.upsert(vectors=vectors)