from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from functools import wraps

import re

app = Flask(__name__)

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per second"])
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
        openai_embed = _get_embedding(oai_client, job_str)
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        cands = _get_shortlist(pc_client, openai_embed, index, k)
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
            soc_desc = re.findall(r"(?<=CGPT587:\s\d{4}\s-\s)(.*?)(?=;\sCONFIDENCE:)", gpt_ans)[0]
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
        soc_followup = "NONE" # Change when refactoring

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

@app.route("/api/adjudicate", methods=["POST"])
@limiter.limit("50 per second")
def adjudicate():
    """
    Endpoint to adjudicate job classifcations from multiple coders.

    For n number of distinct coders, the payload includes the following:
    - adj_prompt: The prompt for the LLM to adjudicate the classifications
    - init_q: The initial question asked to the respondent
    - init_ans: The initial answer provided by the respondent
    - coders: A list of dictionaries, each containing:
        - coder_id: The unique identifier for the coder
        - classification: The classification provided by the coder
        - description: The description of the code from the codebook (optional)
    - include_self_classification: A boolean indicating whether to include the respondent's self-classification
    - model: The model to use for the LLM

    This endpoint provides the option to include or exclude the respondent's self-classification.
    In this case, the respondent's self-classification is included by default.
    If the respondent's self-classification is included, it is added to the list of coders with the identifier "user_soc".
    The LLM will adjudicate the best classification out of n coders, and return the adjudicated classification.

    The response includes:
    - adj_soc_code: The adjudicated SOC code
    - adj_soc_desc: The adjudicated SOC description
    - adj_justification: The justification provided by the LLM for the adjudicated classification
    - adj_hypothesis: The hypothesis provided by the LLM for the discrepancy in classifications
    - coders: A list of dictionaries, each containing:
        - coder_id: The unique identifier for the coder
        - classification: The classification provided by the coder
    - response: The entire response from the LLM 
    """

    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 400
    
    try:
        adj_prompt = data["adj_prompt"]
        init_q = data["init_q"]
        init_ans = data["init_ans"]
        coders = data["coders"]
        include_self_classification = data.get("include_self_classification", True)
        model = data["model"]
    
    except KeyError as e:
        logging.error(f"Missing key in input data: {e}")
        return jsonify({"error": "Missing required fields"}), 400
    
    except Exception as e:
        logging.error(f"Error processing input data: {e}")
        return jsonify({"error": "Invalid input format"}), 400
    
   
    if not isinstance(coders, list) or len(coders) < 2:
        logging.error("Coders must be a list with at least two entries")
        return jsonify({"error": "Coders must be a list with at least two entries"}), 400
   
    for coder in coders:
        if not isinstance(coder, dict) or "coder_id" not in coder or "classification" not in coder:
            logging.error("Each coder must be a dictionary with 'coder_id' and 'classification'")
            return jsonify({"error": "Invalid coder format"}), 400
        
        if "description" not in coder or coder["description"] is None:
            coder["description"] = f"SOC {coder['classification']}"
   
    if include_self_classification:
        user_soc = data.get("user_soc", "NONE")
        user_description = data.get("user_soc_desc", f"SOC {user_soc}") if user_soc != "NONE" else "No self-classification"
        coders.append({
            "coder_id": "respondent",
            "classification": user_soc,
            "description": user_description
            })
    
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        oai_client = OpenAI()
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        return jsonify({"error": "Error calling OpenAI API"}), 500
    

    message_list = [
        {"role": "system", "content": adj_prompt},
        {"role": "assistant", "content": init_q},
        {"role": "user", "content": init_ans},
    ]

    for coder in coders:
        message_list.append(
            {
                "role": "user",
                "content": f"{coder['coder_id']} classified the job as: {coder['classification']} ({coder['description']})",
            }
        )
    message_list.append(
        {
            "role": "user",
            "content": adj_prompt
        }
    )

    completion = oai_client.chat.completions.create(
        model=model,
        messages=message_list,
    )

    gpt_ans = completion.choices[0].message.content

    if gpt_ans.startswith("CGPT587:"):
        try:
            adj_soc_code = re.findall(r"(?<=CGPT587:\s)\d{4}", gpt_ans)[0]
            adj_soc_desc = re.findall(r"(?<=CGPT587:\s\d{4}\s-\s)(.*?)(?=;\sJUSTIFICATION:)", gpt_ans)[0]
            adj_justification = re.findall(r"(?<=JUSTIFICATION:\s)(.*?)(?=;\sHYPOTHESIS:)", gpt_ans)[0]
            adj_hypothesis = re.findall(r"(?<=HYPOTHESIS:\s)(.*?)(?=;\sCODERS:)", gpt_ans)[0]
        except:
            adj_soc_code = "ERROR"
            adj_soc_desc = "ERROR"
            adj_justification = "ERROR"
            adj_hypothesis = "ERROR"
    else:
        adj_soc_code = "NONE"
        adj_soc_desc = "NONE"
        adj_justification = "NONE"
        adj_hypothesis = "NONE"

    return jsonify(
        {
            "adj_soc_code": adj_soc_code,
            "adj_soc_desc": adj_soc_desc,
            "adj_justification": adj_justification,
            "adj_hypothesis": adj_hypothesis,
            "coders": coders,
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
        model=model,
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
