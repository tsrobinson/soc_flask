from flask import Flask, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import hashlib
import os
from functools import wraps, lru_cache
from pathlib import Path

import re

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

app = Flask(__name__)

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per second"])
import logging
from openai import OpenAI
from pinecone import Pinecone

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

KEYRING_SERVICE = "socbot"


def _load_key(env_name: str) -> str | None:
    """Resolve an API key with precedence:

        1. Environment variable (set explicitly, or loaded from .env above)
        2. OS keyring (macOS Keychain / Linux Secret Service / Windows Credential Manager)
        3. None — caller decides whether that's a hard error

    Env wins so CI / Docker / one-shot runs keep working. Keyring is the
    recommended path for local developer machines — see manage_keys.py.
    """
    v = os.environ.get(env_name)
    if v:
        return v
    try:
        import keyring
        v = keyring.get_password(KEYRING_SERVICE, env_name)
        if v:
            return v
    except Exception:
        pass
    return None


## PROVIDER PRE-SETUP
OPENAI_API_KEY = _load_key("OPENAI_API_KEY")
PINECONE_API_KEY = _load_key("PINECONE_API_KEY")
ANTHROPIC_API_KEY = _load_key("ANTHROPIC_API_KEY")
GROQ_API_KEY = _load_key("GROQ_API_KEY")
TOGETHER_API_KEY = _load_key("TOGETHER_API_KEY")

OAI = OpenAI(api_key=OPENAI_API_KEY, timeout=60, max_retries=2) if OPENAI_API_KEY else None
# Groq and Together both expose OpenAI-compatible endpoints. Caller routes to
# them via "groq/<model-id>" or "together/<model-id>" prefixes; we strip the
# prefix and dispatch to the right client.
GROQ = (OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
               timeout=60, max_retries=2) if GROQ_API_KEY else None)
TOGETHER = (OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1",
                   timeout=60, max_retries=2) if TOGETHER_API_KEY else None)
PC = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
_INDEX_CACHE: dict = {}


def _get_index(name: str):
    """Lazily resolve a Pinecone index so startup doesn't fail without keys."""
    if PC is None:
        raise RuntimeError("PINECONE_API_KEY not set; provide 'soc_cands' in the request to skip retrieval")
    if name not in _INDEX_CACHE:
        _INDEX_CACHE[name] = PC.Index(name)
    return _INDEX_CACHE[name]


ANT = (Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120, max_retries=2)
       if (Anthropic and ANTHROPIC_API_KEY) else None)


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
        # Call Pinecone API with timeout (index resolved lazily)
        pc_index = _get_index(index)
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


def _provider_for_model(model: str) -> str:
    m = model.lower()
    if m == "mock":
        return "mock"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("groq/"):
        return "groq"
    if m.startswith("together/"):
        return "together"
    return "openai"


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0}


def _call_openai_compatible(client, sys_prompt, init_q, init_ans, model,
                            extra_turns=None, seed=None):
    """Returns (text, usage_dict). Works for any OpenAI-compatible endpoint
    (real OpenAI, Groq, Together). The seed parameter is best-effort: OpenAI
    accepts and honors it; some compatible endpoints silently ignore it."""
    message_list = [
        {"role": "system", "content": sys_prompt},
        {"role": "assistant", "content": init_q},
        {"role": "user", "content": init_ans},
    ]
    for role, content in (extra_turns or []):
        message_list.append({"role": role, "content": content})
    kwargs = {"model": model, "messages": message_list}
    if seed is not None:
        kwargs["seed"] = int(seed)
    completion = client.chat.completions.create(**kwargs)
    text = completion.choices[0].message.content or ""
    u = completion.usage
    cache_read = 0
    if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
        cache_read = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
        "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": 0,
    }
    return text, usage


def _call_openai(sys_prompt, init_q, init_ans, model, extra_turns=None, seed=None):
    if OAI is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call OpenAI")
    return _call_openai_compatible(OAI, sys_prompt, init_q, init_ans, model,
                                   extra_turns=extra_turns, seed=seed)


def _call_groq(sys_prompt, init_q, init_ans, model, extra_turns=None, seed=None):
    if GROQ is None:
        raise RuntimeError("GROQ_API_KEY not set; cannot call Groq")
    # Strip the "groq/" prefix before sending to Groq's OpenAI-compatible endpoint
    real_model = model.split("/", 1)[1] if model.lower().startswith("groq/") else model
    return _call_openai_compatible(GROQ, sys_prompt, init_q, init_ans, real_model,
                                   extra_turns=extra_turns, seed=seed)


def _call_together(sys_prompt, init_q, init_ans, model, extra_turns=None, seed=None):
    if TOGETHER is None:
        raise RuntimeError("TOGETHER_API_KEY not set; cannot call Together")
    real_model = model.split("/", 1)[1] if model.lower().startswith("together/") else model
    return _call_openai_compatible(TOGETHER, sys_prompt, init_q, init_ans, real_model,
                                   extra_turns=extra_turns, seed=seed)


def _call_anthropic(sys_prompt, init_q, init_ans, model, extra_turns=None,
                    thinking=None, max_tokens=1024):
    """Returns (text, usage_dict). Anthropic requires system as a separate
    param and messages starting with user. The original survey turn
    (interviewer Q → respondent A) is folded into a single user message.
    Any follow-up turns are appended.
    """
    if ANT is None:
        raise RuntimeError("Anthropic SDK not installed or ANTHROPIC_API_KEY missing")
    opening = f"Interviewer: {init_q}\nRespondent: {init_ans}"
    messages = [{"role": "user", "content": opening}]
    for role, content in (extra_turns or []):
        messages.append({"role": role, "content": content})
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": sys_prompt,
        "messages": messages,
    }
    if thinking and thinking != "disabled":
        kwargs["thinking"] = {"type": thinking}
    else:
        kwargs["thinking"] = {"type": "disabled"}
    resp = ANT.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    return text, usage


def _call_mock(sys_prompt, init_q, init_ans, model, extra_turns=None):
    """Deterministic fake response for testing without API keys. Returns
    (text, usage_dict) — usage is a rough char-based estimate."""
    payload = sys_prompt[:200] + "|" + str(init_q) + "|" + str(init_ans)
    h = hashlib.md5(payload.encode()).hexdigest()
    digits = "".join(c for c in h if c.isdigit())
    code = (digits[:4] or "1234").ljust(4, "0")
    alt1 = (digits[4:8] or "2345").ljust(4, "0")
    alt2 = (digits[8:12] or "3456").ljust(4, "0")
    conf = (int(h[:2], 16) % 60) + 40
    text = (f"CGPT587: {code} - Mock occupation; CONFIDENCE: {conf}; "
            f"SHORTLIST: {code}, {alt1}, {alt2}; FOLLOWUP: FALSE; "
            f"JUSTIFICATION: deterministic mock response for model={model};")
    usage = {
        "input_tokens": (len(sys_prompt) + len(init_q) + len(init_ans)) // 4,
        "output_tokens": len(text) // 4,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    return text, usage


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
    provider = _provider_for_model(model)

    if "soc_cands" not in data:
        if provider == "mock":
            cands = "Mock occupation 1 (1111)\nMock occupation 2 (2222)\nMock occupation 3 (3333)"
        else:
            if PC is None:
                return jsonify({"error": "Pinecone not configured; provide 'soc_cands' to skip retrieval"}), 500
            job_str = f"Job title: '{init_ans}'"
            openai_embed = _get_embedding(job_str)
            cands = _get_shortlist(tuple(openai_embed), index, k)
    else:
        cands = data["soc_cands"]

    sys_prompt = sys_prompt.format(**{"K_soc": cands})

    thinking = data.get("thinking")
    max_tokens = int(data.get("max_tokens", 1024))
    conversation = data.get("conversation")
    seed = data.get("seed")
    extra_turns = None
    if conversation:
        extra_turns = [(t["role"], t["content"]) for t in conversation]

    try:
        if provider == "mock":
            gpt_ans, usage = _call_mock(
                sys_prompt, init_q, init_ans, model, extra_turns=extra_turns)
        elif provider == "anthropic":
            # Anthropic doesn't support a seed parameter; if requested we
            # still honor the rest of the call but the seed is a no-op.
            gpt_ans, usage = _call_anthropic(
                sys_prompt, init_q, init_ans, model,
                extra_turns=extra_turns,
                thinking=thinking, max_tokens=max_tokens,
            )
        elif provider == "groq":
            gpt_ans, usage = _call_groq(
                sys_prompt, init_q, init_ans, model,
                extra_turns=extra_turns, seed=seed)
        elif provider == "together":
            gpt_ans, usage = _call_together(
                sys_prompt, init_q, init_ans, model,
                extra_turns=extra_turns, seed=seed)
        else:
            gpt_ans, usage = _call_openai(
                sys_prompt, init_q, init_ans, model,
                extra_turns=extra_turns, seed=seed)
    except Exception as e:
        logging.error(f"{provider} API call failed: {e}")
        return jsonify({"error": f"{provider} API error: {e}"}), 502

    # Accept CGPT587: anywhere in the response. Non-reasoning models (notably
    # Opus 4.6 with thinking disabled) may emit prose reasoning BEFORE the
    # structured line, so startswith() misses it. Search instead.
    #
    # Fields are parsed independently so that one off-format field (e.g.
    # gpt-oss-20b emitting CONFIDENCE: 0.997 or FOLLOWUP: None) doesn't
    # cascade into all-fields-ERROR. The soc_code is the only must-have;
    # everything else is best-effort.
    cgpt_match = re.search(r"CGPT587:\s*\d{4}", gpt_ans)
    if cgpt_match:
        tail = gpt_ans[cgpt_match.start():]

        soc_code_m = re.search(r"(?<=CGPT587:\s)\d{4}", tail)
        soc_code = soc_code_m.group(0) if soc_code_m else "ERROR"

        soc_desc_m = re.search(
            r"(?<=CGPT587:\s\d{4}\s-\s)(.*?)(?=;\s*CONFIDENCE:)", tail
        )
        soc_desc = soc_desc_m.group(0).strip() if soc_desc_m else "ERROR"

        # Accept "85", "0.85", or "85%". Normalize fractions to 0-100.
        conf_m = re.search(r"(?<=CONFIDENCE:\s)([0-9]*\.?[0-9]+)", tail)
        if conf_m:
            try:
                v = float(conf_m.group(1))
                soc_conf = str(int(round(v * 100))) if 0 <= v <= 1 else str(int(round(v)))
            except ValueError:
                soc_conf = "ERROR"
        else:
            soc_conf = "ERROR"

        # Accept TRUE/FALSE/None/null/yes/no — anything else falls back to FALSE.
        fu_m = re.search(r"(?<=FOLLOWUP:\s)([A-Za-z]+)", tail)
        if fu_m:
            v = fu_m.group(1).upper()
            if v in ("TRUE", "YES"):
                soc_followup = "TRUE"
            elif v in ("FALSE", "NO", "NONE", "NULL", "N/A", "NA"):
                soc_followup = "FALSE"
            else:
                soc_followup = "FALSE"
        else:
            soc_followup = "ERROR"
    else:
        soc_code = "NONE"
        soc_desc = "NONE"
        soc_conf = "NONE"
        soc_followup = "NONE"

    return jsonify(
        {
            "soc_code": soc_code,
            "soc_desc": soc_desc,
            "soc_conf": soc_conf,
            "soc_followup": soc_followup,
            "soc_cands": cands.replace("\n", ", "),
            "response": gpt_ans,
            "usage": usage,
            "provider": provider,
            "model": model,
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
