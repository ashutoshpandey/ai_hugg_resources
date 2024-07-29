from transformers import AutoTokenizer, AutoModel
import torch

def get_embedding_dimension(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Sample text
    text = "Sample text to determine embedding dimension"
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    
    # Print the dimension of the embeddings
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    get_embedding_dimension()
