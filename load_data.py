from csv_util import load_csv
from pinecone_util import PineconeUtil
from embedding_util import EmbeddingUtil

embedding_util = EmbeddingUtil()
pinecone_util = PineconeUtil(create_index=True)

# Create text embeddings
def create_embeddings(data):
    embeddings = embedding_util.create_embeddings(data) 
    pinecone_util.store_embeddings(embeddings)

def main():
    print('Loading csv...')
    data = load_csv('data.csv')

    create_embeddings(data)

if __name__ == "__main__":
    main()
