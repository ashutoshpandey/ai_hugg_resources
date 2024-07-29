import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec

def delete_existing_index(index_name):
    load_dotenv()  # Load environment variables from .env file
    pine = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    indexes = pine.list_indexes()
    index_names = [index['name'] for index in indexes.get('indexes', [])]

    if index_name in index_names:
        pine.delete_index(index_name)
        print(f"Deleted index '{index_name}'")

if __name__ == "__main__":
    delete_existing_index('df-resource')
