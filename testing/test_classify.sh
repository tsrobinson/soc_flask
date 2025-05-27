#source ../venv/bin/activate

# Load system prompt
sys_prompt=$(<../classify_prompt.txt)

# Path to job_titles.json
test_inputs="job_titles.json"

# Output directory for results
output_dir="test_results"
mkdir -p "$output_dir"

# Collect already processed job titles
existing_titles=()
if [[ -f "$output_dir/all_results_classify.jsonl" ]]; then
  existing_titles=($(jq -r '.init_ans' "$output_dir/all_results_classify.jsonl"))
fi

# Loop through each job title
jq -c '.[]' "$test_inputs" | while read -r job; do
    if printf '%s\n' "${existing_titles[@]}" | grep -qx "$job"; then
      echo "Skipping already processed: $job"
      continue
    fi
    echo "🔎 Testing job title: $job"

    # Prepare request body
    json_payload=$(jq -n \
      --arg sp "$sys_prompt" \
      --arg q "What was your main job over the last seven days? Please enter your full job title." \
      --arg a "$job" \
      --argjson k 10 \
      --arg index "soc4d" \
      --arg model "o3-mini-2025-01-31" \
      '{sys_prompt: $sp, init_q: $q, init_ans: $a, k: $k, index: $index, model: $model}')
    
    # Try request up to 3 times
    attempts=0
    max_attempts=3
    success=0
    while [[ $attempts -lt $max_attempts ]]; do
      start_time=$(date +%s.%N)
      response=$(curl -s -X POST http://localhost:105/api/classify \
        -H "Content-Type: application/json" \
        -d "$json_payload")
      end_time=$(date +%s.%N)
      elapsed=$(echo "$end_time - $start_time" | bc)
      echo "⏱️  Request for '$job' took ${elapsed}s"

      if echo "$response" | jq -e '.error?' > /dev/null; then
        echo "⚠️ Error for job title: $job (attempt $((attempts+1)))"
        attempts=$((attempts+1))
        sleep 1
      else
        echo "$response" | jq -c --arg init_ans "$job" '. + {init_ans: $init_ans}' >> "$output_dir/all_results_classify.jsonl"
        success=1
        break
      fi
    done

    if [[ $success -eq 0 ]]; then
      echo "$job" >> "$output_dir/failed_jobs.txt"
    fi
    sleep 2
done