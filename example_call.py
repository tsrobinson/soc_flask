import requests

# Define the endpoint URL
url = "https://soc-flask.onrender.com/api/followup"

# Define the JSON payload with placeholder values
payload = {
    "sys_prompt": "followup_prompt.txt",  # Default prompt hard-coded into API
    "init_q": "What is your job title?",
    "init_ans": "Systems developer",
    "index": "soc4d",  # The embeddings index to use
    "k": 30,  # The number of candidate SOC codes to retrieve
    "model": "gpt-4.1-mini-2025-04-14",
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
