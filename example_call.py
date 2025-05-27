import requests

# Define the endpoint URL
url = "https://soc-flask.onrender.com/api/v3"

with open("sys_prompt.txt", "r") as file:
    # Read the content of the file
    sys_prompt = file.read()

# Define the JSON payload with placeholder values
payload = {
    "sys_prompt": sys_prompt,
    "init_q": "What is your job title?",
    "init_ans": "Systems developer",
    "index": "job-titles-4d",  # Replace with your desired index value
    "k": 10,  # Replace with your desired k value
}

# Make the POST request
try:
    response = requests.post(url, json=payload)
    # Check if the request was successful
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response:", response.text)
except Exception as e:
    print("An error occurred:", str(e))
