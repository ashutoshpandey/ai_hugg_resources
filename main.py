import json
import numpy as np
from csv_util import load_csv
from utils.vector.lib.pinecone_util import PineconeUtil
from utils.embeddings.embedding_util import EmbeddingUtil

pinecone_util = PineconeUtil()
embedding_util = EmbeddingUtil('llama')


# Get user input
def get_user_input():
    print('How can I help?')
    query = input()
    return query


# Process the query
def query_database(query, original_data):
    query_embedding = embedding_util.generate_embedding(query)
    result = pinecone_util.process_query(query_embedding)

    process_result(result, original_data, query_embedding)


# Process the response
def process_result(result, original_data, query_embedding, similarity_threshold=0.2):
    json_results = []
    for match in result['matches']:
        user_data = original_data.iloc[int(match['id'])][['name', 'email', 'designation', 'skills']]

        # next 3 lines are for debugging
        user_embedding = embedding_util.generate_embedding(f"{user_data['name']} {user_data['email']} {user_data['designation']} {' '.join(user_data['skills'])}")
        similarity = cosine_similarity(query_embedding, user_embedding)
        print(f"Similarity with '{user_data['name']}': {similarity}")
        
        if similarity >= similarity_threshold:
            json_results.append({
                'name': user_data['name'],
                'email': user_data['email'],
                'designation': user_data['designation'],
                'skills': user_data['skills']
            })

    # Convert list of dictionaries to JSON string
    json_string = json.dumps(json_results, indent=4)
    print(json_string)


# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Program starts here
def main():
    print('Loading csv...')
    data = load_csv('data.csv')
    print(data)

    #print('Generating data')
    #create_embeddings(data)  # Ensure embeddings are created and stored

    while True:
        query = get_user_input()

        if query.lower() == 'exit':
            break  # Correctly exit the loop

        query_database(query, data)

if __name__ == "__main__":
    main()
