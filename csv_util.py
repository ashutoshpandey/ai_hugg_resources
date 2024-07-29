import pandas as pd

# Load CSV file
def load_csv(file_path):
    data = pd.read_csv(file_path)
    data['skills'] = data['skills'].apply(lambda x: x.split(':'))
    return data

