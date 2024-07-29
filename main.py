import pandas as pd  # Corrected import to 'pd' instead of 'panda'
from pinecone_util import PineconeUtil
from embedding_util import EmbeddingUtil

pinecone_util = PineconeUtil()
embedding_util = EmbeddingUtil()

# Load CSV file
def load_csv(file_path):
    data = pd.read_csv(file_path)
    data['skills'] = data['skills'].apply(lambda x: x.split(':'))
    return data


# Get user input
def get_user_input():
    print('How can I help?')
    query = input()
    return query


# Create text embeddings
def create_embeddings(data):
    embeddings = embedding_util.create_embeddings(data)    
    pinecone_util.store_embeddings(embeddings)


# Process the query
def query_database(query, original_data):
    query_embedding = embedding_util.generate_embedding(query)
    
    result = pinecone_util.process_query(query_embedding)

    for match in result['matches']:
        print(original_data.iloc[int(match['id'])][['name', 'email', 'designation', 'skills']])


# Program starts here
def main():
    print('Loading csv...')
    data = load_csv('data.csv')

    print('Generating data')
    create_embeddings(data)  # Ensure embeddings are created and stored

    while True:
        query = get_user_input()

        if query.lower() == 'exit':
            break  # Correctly exit the loop

        query_database(query, data)

if __name__ == "__main__":
    main()
