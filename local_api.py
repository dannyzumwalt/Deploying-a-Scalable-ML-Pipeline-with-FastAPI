import json
import requests

# Send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# Print the status code
print("GET status code:", r.status_code)
# Print the welcome message
print("GET response:", r.json())


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data)

#  print the status code
print("POST status code:", r.status_code)
#  print the result
print("POST response:", r.json())