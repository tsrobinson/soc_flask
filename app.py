from flask import Flask, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from functools import wraps, lru_cache
from pathlib import Path

import re

app = Flask(__name__)

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per second"])
import logging
from openai import OpenAI
from pinecone import Pinecone

## OPEN AI AND PINECONE PRE-SETUP
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

OAI = OpenAI(timeout=10, max_retries=2)  # 10 s global timeout
PC = Pinecone(api_key=PINECONE_API_KEY)  # keep tcp-pool alive
# INDEX = PC.Index("job-titles-4d")
INDEXES = {
    "job-titles-4d": PC.Index("job-titles-4d"),
    "soc4d": PC.Index("soc4d"),
}


## PROMPT PRE-LOADING
PROMPTS = {
    p.name: p.read_text("utf-8")
    for p in (Path(__file__).parent / "static").glob("*.txt")
}

# Setup logging
logging.basicConfig(level=logging.INFO)


@app.route("/hello/", methods=["GET", "POST"])
def welcome():
    return "Hello World!"


def get_followup_prompt(prompt: str) -> str:
    return PROMPTS[prompt]


def check_input(data):

    try:
        sys_prompt = data["sys_prompt"]
        init_q = data["init_q"]
        init_ans = data["init_ans"]
    except KeyError:
        logging.error("Invalid input")
        return jsonify({"error": "Invalid input"}), 400

    # Handle case where .txt path provided instead of full text
    if sys_prompt[-4:] == ".txt":
        try:
            sys_prompt = get_followup_prompt(sys_prompt)
        except FileNotFoundError:
            logging.error(f"System prompt file '{sys_prompt}' not found")
            return (
                jsonify({"error": f"System prompt file '{sys_prompt}' not found"}),
                400,
            )

    return sys_prompt, init_q, init_ans


@lru_cache(maxsize=2_000)
def _get_embedding(text, model="text-embedding-3-large", dimensions=3072):
    try:
        # Call OpenAI API with timeout
        return (
            OAI.embeddings.create(input=[text], model=model, dimensions=dimensions)
            .data[0]
            .embedding
        )
    except Exception as e:
        logging.error(f"OpenAI embeddings API call failed: {e}")
        return (
            jsonify({"error": "Error calling OpenAI embeddings API"}),
            500,
        )


@lru_cache(maxsize=2_000)
def _get_shortlist(embedding: tuple[float, ...], index: str, k: int):
    try:
        # Call Picone API with timeout
        pc_index = INDEXES[index]
        pinecone_response = pc_index.query(
            vector=embedding, top_k=k, include_metadata=True
        )

        # Check if the response is valid
        if not pinecone_response:
            raise ValueError("Empty response from Pinecone API")

        results = pinecone_response.matches

        if index == "job-titles-4d":
            # extracts code descriptior (code) for each match, then removes any duplicates
            unique_cands = list(
                dict.fromkeys([result["metadata"]["desc"] for result in results])
            )
            cands = "".join([code + "\n" for code in unique_cands])
        else:
            cands = "".join([result.id + "\n" for result in results])

        return cands

    except Exception as e:
        logging.error(f"Pinecone API call failed: {e}")
        return jsonify({"error": "Error calling Pinecone API"}), 500


@app.route("/api/classify", methods=["POST"])
@limiter.limit("50 per second")
def classify():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        k = int(data["k"])
        index = data["index"]
        model = data["model"]
    except Exception as e:
        return jsonify({"error": "Missing or invalid 'k', 'index', or 'model'"}), 400

    sys_prompt, init_q, init_ans = check_input(data)

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        oai_client = OpenAI()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return jsonify({"error": "Error calling OpenAI API"}), 500

    if "soc_cands" not in data:
        job_str = f"Job title: '{init_ans}'"
        openai_embed = _get_embedding(job_str)
        # pc_client = Pinecone(api_key=PINECONE_API_KEY)
        cands = _get_shortlist(tuple(openai_embed), index, k)
    else:
        cands = data["soc_cands"]

    sys_prompt = sys_prompt.format(**{"K_soc": cands})

    message_list = [
        {"role": "system", "content": sys_prompt},
        {"role": "assistant", "content": init_q},
        {"role": "user", "content": init_ans},
    ]

    completion = oai_client.chat.completions.create(
        model=model,
        messages=message_list,
    )

    gpt_ans = completion.choices[0].message.content
    if gpt_ans.startswith("CGPT587:"):
        try:
            soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", gpt_ans)[0]
            soc_desc = re.findall(
                r"(?<=CGPT587:\s\d{4}\s-\s)(.*?)(?=;\sCONFIDENCE:)", gpt_ans
            )[0]
            soc_conf = re.findall(r"(?<=CONFIDENCE:\s)\d+", gpt_ans)[0]
            soc_followup = re.findall(r"(?<=FOLLOWUP:\s)(TRUE|FALSE)", gpt_ans)[0]
        except:
            soc_code = "ERROR"
            soc_desc = "ERROR"
            soc_conf = "ERROR"
            soc_followup = "ERROR"

    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"
        soc_followup = "NONE"  # Change when refactoring

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "soc_followup": soc_followup,
            "soc_cands": cands.replace("\n", ", "),
            "response": gpt_ans,
        }
    )


@app.route("/api/followup", methods=["POST"])
@limiter.limit("50 per second")
def followup():

    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        k = int(data["k"])
        index = data["index"]
        model = data["model"]
    except Exception as e:
        return jsonify({"error": "Invalid input for either k, index, or model"}), 400

    sys_prompt, init_q, init_ans = check_input(data)

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # oai_client = OpenAI()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return jsonify({"error": "Error calling OpenAI API"}), 500

    if "soc_cands" not in data:
        job_str = f"Job title: '{init_ans}'"
        openai_embed = _get_embedding(job_str)
        # pc_client = Pinecone(api_key=PINECONE_API_KEY)
        cands = _get_shortlist(tuple(openai_embed), index, k)

    elif data["soc_cands"] == "REQUERY":
        job_str = f"Job title: '{data['additional_qs'][0][1]}'"
        openai_embed = _get_embedding(job_str)
        # pc_client = Pinecone(api_key=PINECONE_API_KEY)
        cands = _get_shortlist(tuple(openai_embed), index, k)

    else:
        cands = data["soc_cands"]

    sys_prompt = sys_prompt.format(**{"K_soc": cands})

    message_list = []

    message_list.append({"role": "system", "content": sys_prompt})
    message_list.append({"role": "assistant", "content": init_q})
    message_list.append({"role": "user", "content": init_ans})

    if "additional_qs" in data:
        for add_q, add_ans in data["additional_qs"]:
            message_list.append({"role": "assistant", "content": add_q})
            message_list.append({"role": "user", "content": add_ans})

    completion = OAI.chat.completions.create(
        model=model,
        messages=message_list,
        temperature=0.01,
        top_p=1.0,
    )

    gpt_ans = completion.choices[0].message.content
    if len(re.findall("CGPT587", gpt_ans)) > 0:
        try:
            soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", gpt_ans)[0]
            soc_desc = re.findall(
                r"(?<=CGPT587:\s\d{4}\s-\s).*(?=\s\(\d+\)$)", gpt_ans
            )[0]
            soc_conf = re.findall(r"\d+(?=\)$)", gpt_ans)[0]
        except:
            soc_code = "ERROR"
            soc_desc = "ERROR"
            soc_conf = "ERROR"
    elif len(re.findall("CGPT631:", gpt_ans)) > 0:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"
        cands = "REQUERY"
        gpt_ans = re.sub("CGPT631: ", "", gpt_ans)
    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "followup": gpt_ans,
            "soc_cands": cands,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105)
