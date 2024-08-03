from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        embedding = self.model.encode(text)
        assert len(embedding) == 384, f"Embedding for '{text}' has incorrect dimension: {len(embedding)}"
        return embedding

    def create_embeddings(self, data):
        data['embedding'] = data.apply(lambda row: self.generate_embedding(f"{row['name']} {row['email']} {row['designation']} {' '.join(row['skills'])}"), axis=1)
        return data
