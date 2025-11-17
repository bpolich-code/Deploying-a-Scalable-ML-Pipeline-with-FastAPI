import requests

base_url = 'http://127.0.0.1:8000'

print('Testing GET request...')
response = requests.get(base_url + '/')
print(f'Status Code: {response.status_code}')
print(f'Result: {response.json()}')
print()

print('Testing POST request...')
data = {
    'age': 37,
    'workclass': 'Private',
    'fnlgt': 178356,
    'education': 'HS-grad',
    'education-num': 10,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Prof-specialty',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

response = requests.post(base_url + '/predict', json=data)
print(f'Status Code: {response.status_code}')
print(f'Result: {response.json()}')
