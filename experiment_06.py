from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

documents = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Retrieval-Augmented Generation combines document retrieval with text generation.",
    "Python is a popular high-level programming language used in AI development.",
    "Vector databases store embeddings and support fast similarity search."
]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

query = "What is RAG in AI?"
query_embedding = embed_model.encode([query])

D, I = index.search(np.array(query_embedding), k=2)
retrieved_chunks = [documents[i] for i in I[0]]

context = " ".join(retrieved_chunks)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

answer = generator(prompt, max_length=60)

print("Retrieved Context:", retrieved_chunks)
print("Answer:", answer[0]["generated_text"])
