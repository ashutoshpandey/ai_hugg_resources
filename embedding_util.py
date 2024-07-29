import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingUtil:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        
        return embeddings.squeeze().numpy()


    def create_embeddings(self, data):
        data['embedding'] = data.apply(lambda row: self.generate_embedding(f"{row['name']} {row['email']} {row['designation']} {' '.join(row['skills'])}"), axis=1)
        return data
