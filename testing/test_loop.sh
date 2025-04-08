#!/bin/bash

# Load system prompt
sys_prompt=$(<../sys_prompt.txt)

# Path to job_titles.json
test_inputs="job_titles.json"

# Output directory for results
output_dir="test_results"
mkdir -p "$output_dir"

# Loop through each job title
jq -c '.[]' "$test_inputs" | while read -r job; do
    echo "🔍 Testing job title: $job"

    # Prepare request body
    json_payload=$(jq -n \
      --arg sp "$sys_prompt" \
      --arg q "What was your main job over the last seven days? Please enter your full job title." \
      --arg a "$job" \
      '{sys_prompt: $sp, init_q: $q, init_ans: $a}')

    # Send curl request and save response
    response=$(curl -s -X POST http://localhost:105/api/v2 \
      -H "Content-Type: application/json" \
      -d "$json_payload")

    # Save response to file
    echo "$response" | jq -c '.' >> "$output_dir/all_results.jsonl"
    sleep 6.5
done