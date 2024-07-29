import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec

class PineconeUtil:
    def __init__(self):
        self.index_name = 'df-resource'
        self.initialize_pinecone()

    # Initialize Pinecone
    def initialize_pinecone(self):
        load_dotenv()  # Load environment variables from .env file
        self.pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        if self.index_name in self.pinecone.list_indexes().names():
            print('Deleting existing index')
            self.pinecone.delete_index(self.index_name)

        print('Creating index...')
        self.pinecone.create_index(
            name=self.index_name,
            dimension=384,  # Correct dimension for the embedding model used
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

        # wait for index to be initialized
        while not self.pinecone.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        print(f"Created index '{self.index_name}'")

    # Store embeddings in Pinecone
    def store_embeddings(self, embeddings):
        index = self.pinecone.Index(self.index_name)
        batch_size = 100  # Define a batch size for upserting

        vectors = [{"id": str(i), "values": row['embedding'].tolist()} for i, row in embeddings.iterrows()]
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)

    # Process the query
    def process_query(self, embedding):
        index = self.pinecone.Index(self.index_name)
        query_embedding = embedding.tolist()  # Convert numpy array to list
        result = index.query(queries=[{"values": query_embedding}], top_k=5)
        return result
