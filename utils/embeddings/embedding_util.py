from utils.embeddings.lib.hugging_face import HuggingFacembedding
from utils.embeddings.lib.sentence_transformer import SentenceTransformerEmbedding


class EmbeddingUtil:
    def __init__(self, model = 'huggingface'):
        self.model = model


    def generate_embedding(self, text):
        if self.model == 'huggingface':
            embedding = HuggingFacembedding()
            return embedding.generate_embedding(text)
        elif self.model == 'sentance':
            embedding = SentenceTransformerEmbedding()
            return embedding.generate_embedding(text)
        elif self.model == 'llama':
            embedding = SentenceTransformerEmbedding()
            return embedding.generate_embedding(text)


    def create_embeddings(self, data):
        data['embedding'] = data.apply(lambda row: self.generate_embedding(f"{row['name']} {row['email']} {row['designation']} {' '.join(row['skills'])}"), axis=1)
        return data
