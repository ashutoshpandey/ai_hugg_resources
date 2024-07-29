import json
from csv_util import load_csv
from pinecone_util import PineconeUtil
from embedding_util import EmbeddingUtil

pinecone_util = PineconeUtil()
embedding_util = EmbeddingUtil()


# Get user input
def get_user_input():
    print('How can I help?')
    query = input()
    return query


# Process the query
def query_database(query, original_data):
    query_embedding = embedding_util.generate_embedding(query)
    result = pinecone_util.process_query(query_embedding)
    print(result)
    process_result(result, original_data)


# Process the response
def process_result(result, original_data):
    json_results = []
    for match in result['matches']:
        user_data = original_data.iloc[int(match['id'])][['name', 'email', 'designation', 'skills']]
        json_results.append({
            'name': user_data['name'],
            'email': user_data['email'],
            'designation': user_data['designation'],
            'skills': user_data['skills']
        })

    # Convert list of dictionaries to JSON string
    json_string = json.dumps(json_results, indent=4)
    print(json_string)


# Program starts here
def main():
    print('Loading csv...')
    data = load_csv('data.csv')

    #print('Generating data')
    #create_embeddings(data)  # Ensure embeddings are created and stored

    while True:
        query = get_user_input()

        if query.lower() == 'exit':
            break  # Correctly exit the loop

        query_database(query, data)

if __name__ == "__main__":
    main()
