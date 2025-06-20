import requests
import re

# Define the endpoint URL
# url = "https://soc-flask.onrender.com/api/followup"
url = "http://127.0.0.1:105/api/followup"  # For local testing

if __name__ == "__main__":

    init_q = "What is your job title?"
    init_ans = input(init_q + " ")

    # Make the POST request
    # Define the JSON payload with placeholder values
    payload = {
        "sys_prompt": "followup_prompt.txt",  # Default prompt hard-coded into API
        "init_q": init_q,
        "init_ans": init_ans,
        "index": "soc4d",  # The embeddings index to use
        "k": 30,  # The number of candidate SOC codes to retrieve
        "model": "gpt-4o-mini-2024-07-18",
    }
    response = ""
    additional_qs = []
    while not response or len(re.findall("CGPT587", response.json()["followup"])) == 0:
        try:
            response = requests.post(url, json=payload)
            # Check if the request was successful
            if response.status_code == 200:
                print("Response:", response.json())

                if len(re.findall("CGPT587", response.json()["followup"])) > 0:
                    print(response.json()["followup"])
                    break

                new_response = input(f"\n {response.json()['followup']} ")
                additional_qs.append([response.json()["followup"], new_response])
                payload["additional_qs"] = additional_qs
                payload["soc_cands"] = response.json()["soc_cands"]
            else:
                print(f"Failed with status code: {response.status_code}")
                print("Response:", response.text)
                break
        except Exception as e:
            print("An error occurred:", str(e))
            break
