import os
import re

import requests

# Define the endpoint URL
url = os.getenv("SOCBOT_URL", "http://localhost:105/api/followup")

# Access token for the service. This is issued by whoever runs the API; it is
# not an OpenAI or Pinecone key.
TOKEN = os.getenv("SOCBOT_API_TOKEN")
if not TOKEN:
    raise SystemExit("Set SOCBOT_API_TOKEN to the token issued for this service.")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

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

    # initialise store of additional questions and answers
    response = ""
    additional_qs = []

    # continue to query API until classificationm made
    while not response or len(re.findall("CGPT587", response.json()["followup"])) == 0:
        try:
            response = requests.post(url, json=payload, headers=HEADERS)
            # Check if the request was successful
            if response.status_code == 200:
                print("Response:", response.json())

                # break if classification made
                if len(re.findall("CGPT587", response.json()["followup"])) > 0:
                    print(response.json()["followup"])
                    break

                # otherwise, ask follow-up question
                new_response = input(f"\n {response.json()['followup']} ")
                additional_qs.append([response.json()["followup"], new_response])
                payload["additional_qs"] = additional_qs

                # make sure to update soc_cands EACH ITERATION, in case it has changed
                payload["soc_cands"] = response.json()["soc_cands"]
            else:
                print(f"Failed with status code: {response.status_code}")
                print("Response:", response.text)
                break
        except Exception as e:
            print("An error occurred:", str(e))
            break
