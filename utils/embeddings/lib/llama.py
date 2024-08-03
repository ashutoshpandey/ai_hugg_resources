from transformers import AutoTokenizer, AutoModel
import torch

class LLamaEmbeddingUtil:
    def __init__(self, model_name='meta-llama/LLaMA-2-7b', use_gpu=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.model.to(self.device)

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        assert len(embeddings) == 4096, f"Embedding for '{text}' has incorrect dimension: {len(embeddings)}"
        return embeddings

    def create_embeddings(self, data):
        data['embedding'] = data.apply(lambda row: self.generate_embedding(f"{row['name']} {row['email']} {row['designation']} {' '.join(row['skills'])}"), axis=1)
        return data
