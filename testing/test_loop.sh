#!/bin/bash

# Load system prompt
sys_prompt=$(<../sys_prompt.txt)

# Path to job_titles.json
test_inputs="job_titles.json"

# Output directory for results
output_dir="test_results"
mkdir -p "$output_dir"

# Collect already processed job titles
existing_titles=()
if [[ -f "$output_dir/all_results.jsonl" ]]; then
  existing_titles=($(jq -r '.init_ans' "$output_dir/all_results.jsonl"))
fi

# Loop through each job title
jq -c '.[]' "$test_inputs" | while read -r job; do
    if printf '%s\n' "${existing_titles[@]}" | grep -qx "$job"; then
      echo "⏭️  Skipping already processed: $job"
      continuea
    fi
    echo "🔍 Testing job title: $job"

    # Prepare request body
    json_payload=$(jq -n \
      --arg sp "$sys_prompt" \
      --arg q "What was your main job over the last seven days? Please enter your full job title." \
      --arg a "$job" \
      '{sys_prompt: $sp, init_q: $q, init_ans: $a}')

    # Try request up to 3 times
    attempts=0
    max_attempts=3
    success=0
    while [[ $attempts -lt $max_attempts ]]; do
      response=$(curl -s -X POST http://localhost:105/api/v2 \
        -H "Content-Type: application/json" \
        -d "$json_payload")

      if echo "$response" | jq -e '.error?' > /dev/null; then
        echo "⚠️  Error for job title: $job (attempt $((attempts+1)))"
        attempts=$((attempts+1))
        sleep 3
      else
        echo "$response" | jq -c --arg init_ans "$job" '. + {init_ans: $init_ans}' >> "$output_dir/all_results.jsonl"
        success=1
        break
      fi
    done

    if [[ $success -eq 0 ]]; then
      echo "$job" >> "$output_dir/failed_jobs.txt"
    fi
    sleep 6.5
done