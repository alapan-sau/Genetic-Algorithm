import json
import requests
import numpy as np

API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11

def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

def get_errors(id, vector):
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))

def get_overfit_vector(id):
    return json.loads(send_request(id, [0], 'getoverfit'))

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    print(get_errors('SECRET_KEY', get_overfit_vector('SECRET_KEY')))
    print(get_overfit_vector('SECRET_KEY'))
    print(submit('SECRET_KEY', get_overfit_vector('SECRET_KEY')))



