# SOC coding flask container

This is a Flask app that sets up a custom API to perform retrieval augmented generation of SOC codes, based on survey responses. 

## Endpoint

The current endpoint is `/api/v2`, which accepts a POST request with JSON payload. The JSON payload should have `init_q` and `init_ans` strings, and an optional `prompt` string that contains instructions to the LLM (otherwise it will read the system prompt in this directory.

In subsequent calls, you can append a JSON object containing question-answer pairs as lists.

## Render

This app is being served via Render. Requests are limited.