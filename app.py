from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from functools import wraps

import re

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
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        init_info = data["init_info"]
    except KeyError:
        return jsonify({"error": "Invalid input"}), 400

    text = init_info

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

    if "soc_cands" not in data:
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

    else:
        ids = data["soc_cands"]

    # Query ChatGPT using the prompt (if no prompt provided as input, use the default prompt)
    if "prompt" not in data:
        with open("prompt.txt", "r") as f:
            prompt = f.read()
    else:
        prompt = data["prompt"]

    if "additional_info" not in data:
        additional_info = "None"
    else:
        additional_info = data["additional_info"]

    prompt = prompt.format(
        **{
            "init_info": init_info,
            "K_soc": ids,
            "additional_info": additional_info,
        }
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": prompt},
        ],
    )

    gpt_ans = completion.choices[0].message.content
    if len(re.findall("CGPT587", gpt_ans)) > 0:
        try:
            soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}\/\d{2}", gpt_ans)[0]
            soc_desc = re.findall(r"(?<=-\s).*(?=\s\(\d+\)$)", gpt_ans)[0]
            soc_conf = re.findall(r"\d+(?=\)$)", gpt_ans)[0]
        except:
            soc_code = "ERROR"
            soc_desc = "ERROR"
            soc_conf = "ERROR"
    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "followup": completion.choices[0].message.content,
            "soc_cands": ids,
        }
    )


def check_input(data):
    try:
        sys_prompt = data["sys_prompt"]
        init_q = data["init_q"]
        init_ans = data["init_ans"]
    except KeyError:
        logging.error("Invalid input")
        return jsonify({"error": "Invalid input"}), 400

    return sys_prompt, init_q, init_ans


def _get_embedding(client, text, model="text-embedding-3-large", dimensions=3072):
    try:
        # Call OpenAI API with timeout
        return (
            client.embeddings.create(input=[text], model=model, dimensions=dimensions)
            .data[0]
            .embedding
        )
    except Exception as e:
        logging.error(f"OpenAI embeddings API call failed: {e}")
        return (
            jsonify({"error": "Error calling OpenAI embeddings API"}),
            500,
        )


def _get_shortlist(client, embedding, index, k):
    try:
        # Call Picone API with timeout
        pc_index = client.Index(index)
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


@app.route("/api/v2", methods=["POST"])
@limiter.limit("10 per minute")  # Rate limit for this endpoint
def v2():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        sys_prompt = data["sys_prompt"]
        init_q = data["init_q"]
        init_ans = data["init_ans"]
    except KeyError:
        return jsonify({"error": "Invalid input"}), 400

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        client = OpenAI()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return jsonify({"error": "Error calling OpenAI API"}), 500

    if "soc_cands" not in data:

        try:
            # Call OpenAI API with timeout
            openai_response = (
                client.embeddings.create(
                    input=[init_ans], model="text-embedding-3-large", dimensions=3072
                )
                .data[0]
                .embedding
            )
        except Exception as e:
            logging.error(f"OpenAI embeddings API call failed: {e}")
            return (
                jsonify({"error": "Error calling OpenAI embeddings API"}),
                500,
            )

        try:
            # Call Picone API with timeout
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index("soc-v2")
            pinecone_response = index.query(vector=openai_response, top_k=10)

            # Check if the response is valid
            if not pinecone_response:
                raise ValueError("Empty response from Pinecone API")

        except Exception as e:
            logging.error(f"Pinecone API call failed: {e}")
            return jsonify({"error": "Error calling Pinecone API"}), 500

        results = pinecone_response.matches
        cands = "".join([result.id + "\n" for result in results])

    else:
        cands = data["soc_cands"]

    print(cands)

    sys_prompt = sys_prompt.format(**{"K_soc": cands})

    message_list = []

    message_list.append({"role": "system", "content": sys_prompt})
    message_list.append({"role": "assistant", "content": init_q})
    message_list.append({"role": "user", "content": init_ans})

    if "additional_qs" in data:
        for add_q, add_ans in data["additional_qs"]:
            message_list.append({"role": "assistant", "content": add_q})
            message_list.append({"role": "user", "content": add_ans})

    completion = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=message_list,
    )

    gpt_ans = completion.choices[0].message.content
    if len(re.findall("CGPT587", gpt_ans)) > 0:
        try:
            soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", gpt_ans)[0]
            soc_desc = re.findall(
                r"(?<=CGPT587:\s\d{4}\/\d{2}\s-\s).*(?=\s\(\d+\)$)", gpt_ans
            )[0]
            soc_conf = re.findall(r"\d+(?=\)$)", gpt_ans)[0]
        except:
            soc_code = "ERROR"
            soc_desc = "ERROR"
            soc_conf = "ERROR"
    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "followup": completion.choices[0].message.content,
            "soc_cands": cands,
        }
    )


def _extract_codes(shortlist):
    codes = []
    for item in shortlist:
        codes.append(item["id"])
    return codes


@app.route("/api/v3", methods=["POST"])
@limiter.limit("10 per minute")  # Rate limit for this endpoint
def v3():

    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        k = int(data["k"])
        index = data["index"]
    except Exception as e:
        return jsonify({"error": "Invalid input for either k or index"}), 400

    sys_prompt, init_q, init_ans = check_input(data)

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        oai_client = OpenAI()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return jsonify({"error": "Error calling OpenAI API"}), 500

    if "soc_cands" not in data:

        job_str = f"Job title: '{init_ans}'"
        openai_embed = _get_embedding(oai_client, job_str)
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        cands = _get_shortlist(pc_client, openai_embed, index, k)

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

    completion = oai_client.chat.completions.create(
        # model="gpt-4o-2024-11-20",
        model="o3-mini-2025-01-31",
        messages=message_list,
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
    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "followup": completion.choices[0].message.content,
            "soc_cands": cands,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=105)
