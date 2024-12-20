# SOC coding flask container

This is a Flask app that sets up a custom API to perform retrieval augmented generation of SOC codes, based on survey responses. 

## Endpoint

The endpoint is `/api/get_result` and it accepts a POST request with a JSON payload. The JSON payload should have `job_title`, `job_description` and `employer_industry` strings, a `text` string (normally the concatenation of the three previous fields), and an optional `prompt` string that contains instructions to the LLM. 

If no `prompt` is provided, the default prompt in this directory is used.

## Render

This app is being served via Render, on the free tier. Requests are limited.