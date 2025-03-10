# SOC-Classifier API Documentation

## /hello/ Endpoint

### Description

A simple endpoint to test the server is up, returning a greeting message. It supports both GET and POST requests.

URL: /hello/

### HTTP Methods

GET
POST

### Request Parameters
None

### Request Examples
GET Request:

```
GET /hello/ HTTP/1.1
Host: yourserver.com
POST Request:
```

### Response

*200 OK:*
    Returns a plain text response:


## /api/v3 Endpoint

### Description

The /api/v3 endpoint processes a POST request to determine SOC (Standard Occupational Classification) details. It leverages OpenAI to generate an embedding from a candidate's job title then queries a vector database (hosted on Pinecone) to retrieve the closest candidate SOC IDs. A conversation prompt is then constructued to obtain the final SOC classification from an AI model. There are two possible returns: a SOC code (where the return is prefixed with 'CGPT587: ') or a followup question.

### URL
/api/v3

### HTTP Methods
POST

### Rate Limiting
Limited to 10 requests per minute (currently).

### Request Parameters

**JSON Body (Required Fields)**

  * k (integer): The number of top candidate results to retrieve from the Pinecone API.
  
  * index (string): The name of the Pinecone index to query.
  
  * sys_prompt (string): A system prompt template. It must include the placeholder {K_soc}, which will be replaced with candidate SOC IDs.
  
  * init_q (string): The initial query message used as the assistant’s message.
  
  * init_ans (string): The initial answer or input text used both for generating embeddings and as the user’s message.

**JSON Body (Optional Fields)**

  * soc_cands (string): Pre-provided candidate SOC IDs. If provided, the endpoint bypasses the Pinecone lookup (useful when posing a followup question)
  
  * *additional_qs (array): An array of additional question-answer pairs. Each pair should be provided as a tuple where:
    * The first element is the previous followup message (question).
    * The second element is a user’s response (answer).


### Request Example

```json
{
  "k": 10,
  "index": "soccode-index",
  "sys_prompt": "Please classify the occupation of this subject from the following options:\n{K_soc}",
  "init_q": "In the past week, what was your main job title?",
  "init_ans": "I was a software developer",
  "soc_cands": "Optional: candidate IDs if already available from a previous call.",
  "additional_qs": [
    ["Could you refine the classification?", "Additional context for clarification."],
    ["After that additional context, could you tell me more?", "No, I am afraid I can't"]
  ]
}
```

### Processing Flow

*OpenAI Embedding and Pinecone Query:*
If the soc_cands field is not provided, the service generates an embedding for init_ans using the OpenAI API. This embedding is used to query the specified Pinecone index with the top k results. The candidate SOC IDs are collected as a newline-separated string.

*Prompt Formation:*
The sys_prompt is formatted by replacing the {K_soc} placeholder with the candidate SOC IDs.

*Message List Construction:*
A conversation is constructed in the following order:

  * System Message: The formatted sys_prompt
  * Assistant Message: init_q
  * User Message: init_ans
  * Additional Messages: (if provided) each additional question-answer pair is appended to the conversation.

This conversation is sent to the OpenAI chat completions API using model gpt-4o-2024-11-20 to generate a response.

*Response Parsing:*
The GPT response is scanned for a specific marker (i.e., "CGPT587"). If found, regular expressions extract:

  * soc_code: A four-digit code.
  * soc_desc: The SOC description.
  * soc_conf: A confidence score.

If the marker is not found, these fields default to "NONE". 

If an error occurs during extraction, they are set to "ERROR".

*JSON Response:*
The endpoint returns a JSON object with the extracted SOC details, the full GPT response, and the candidate SOC IDs.

### Response

*Success Response (HTTP 200)*

```json
{
  "soc_code": "1234",          // Extracted SOC code, or "NONE" if not found.
  "soc_desc": "Occupation Description", // Extracted SOC description, or "NONE" if not found.
  "soc_conf": "95",            // Extracted confidence score, or "NONE" if not found.
  "followup": "CGPT587: 1234 - Occupation Description (95)",
  "soc_cands": "1234\n5678\n..."  // Newline-separated candidate SOC IDs.
}
```

Error Responses are returned for failed external API calls to Pinecone or OpenAI.

### Additional Notes

*Environment Variables:*
Ensure that the OPENAI_API_KEY and PINECONE_API_KEY environment variables are set correctly.

*System Prompt Template:*
The sys_prompt must include the placeholder {K_soc} so that candidate SOC IDs can be dynamically inserted.

*Optional Candidate Bypass:*
If you already have candidate SOC IDs, you can provide them via the soc_cands field to bypass the embedding and Pinecone query process.

*Additional Context:*
You may extend the conversation context by including additional question-answer pairs in the additional_qs field.

*Looping requests:*
It is imagined that the endpoint will be used in a loop, with some limit to the number of followups and early termination if a code is returned. After each API response, if a followup question is returned this question and the subject's answer should be appended to the next call through the `additional_qs` argument. If the maximum number of followups is reached (set outside the API), the final sys_prompt should be amended to demand a code rather than allow for the option of a followu question.