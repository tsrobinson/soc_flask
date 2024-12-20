from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from functools import wraps

app = Flask(__name__)

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour"])
import logging
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)


@app.route("/hello/", methods=["GET", "POST"])
def welcome():
    return "Hello World!"


@app.route("/api/get_results", methods=["POST"])
@limiter.limit("10 per minute")  # Rate limit for this endpoint
def get_results():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Invalid input"}), 400

    text = data["text"]

    # Input validation
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text must be a non-empty string"}), 400

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        client = OpenAI()
        # Call OpenAI API with timeout
        text = text.replace("\n", " ")
        openai_response = (
            client.embeddings.create(
                input=[text], model="text-embedding-3-small", dimensions=512
            )
            .data[0]
            .embedding
        )
    except Exception as e:
        return jsonify({"error": f"Error calling OpenAI API: {str(e)}"}), 500

    try:
        # Call Picone API with timeout
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("soccode-index")
        pinecone_response = index.query(vector=openai_response, top_k=10)

        # Check if the response is valid
        if not pinecone_response:
            raise ValueError("Empty response from Pinecone API")

    except Exception as e:
        logging.error(f"Pinecone API call failed: {e}")
        return jsonify({"error": "Error calling Pinecone API"}), 500

    results = pinecone_response.matches
    ids = ""
    for result in results:
        ids += result.id + "\n"
    ids = ids[:-2]

    # Query ChatGPT using the prompt (if no prompt provided as input, use the default prompt)
    if "prompt" not in data:
        with open("prompt.txt", "r") as f:
            prompt = f.read()
    else:
        prompt = data["prompt"]

    try:
        empl_ind = data["employer_industry"]
        job_title = data["job_title"]
        job_desc = data["job_description"]
    except KeyError:
        return jsonify({"error": "Invalid input"}), 400

    prompt = prompt.format(
        **{
            "employer_industry": empl_ind,
            "job_title": job_title,
            "job_description": job_desc,
            "K_soc": ids,
        }
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": prompt},
        ],
    )

    return jsonify({"followup": completion.choices[0].message.content})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105)
