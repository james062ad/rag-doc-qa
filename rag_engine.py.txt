import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

class RAGPipeline:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', llm_model='gpt-3.5-turbo', api_key=None):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_model = llm_model
        openai.api_key = api_key or openai.api_key
        self.chunks = []
        self.index = None

    def chunk_text(self, text, chunk_size=100, overlap=20):
        words = text.split()
        self.chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            self.chunks.append(" ".join(words[start:end]))
            if end >= len(words): break
            start = end - overlap
        return self.chunks

    def embed_chunks(self):
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        return embeddings

    def build_index(self, embeddings):
        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def process(self, text):
        self.chunk_text(text)
        embeddings = self.embed_chunks()
        self.build_index(embeddings)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def answer(self, question, top_k=5):
        context_chunks = self.retrieve(question, top_k)
        context = " ".join(context_chunks)
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
