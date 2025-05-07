import pandas as pd
import asyncio
import httpx
import json
import math
from typing import List

API_URL = "http://localhost:105/api/classify"
BATCH_SIZE = 45  
TIMEOUT = 50.0 

### ALWAYS CHECK BEFORE RUNNING ###
with open("../classify_prompt.txt", "r") as f:
    SYS_PROMPT = f.read()

df = pd.read_csv("PATH_TO_CSV")

JOB_TITLES = [row['title_industry'] for _, row in df.iterrows()]

true_map = {
    row["title_industry"]: {"id": row["id"], "soc_true": row["soc"]}
    for _, row in df.iterrows()
}
### ALWAYS CHECK BEFORE RUNNING ###

async def classify(client: httpx.AsyncClient, job_title: str):
    payload = {
        "sys_prompt": SYS_PROMPT,
        "init_q": "What was your main job over the last seven days? Please enter your job title and the industry you are in.",
        "init_ans": job_title,
        "k": 10,
        "index": "soc4d",
        "model": "o4-mini-2025-04-16"
    }

    try:
        response = await client.post(API_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()
        print(f"{job_title} → {result.get('soc_code', 'N/A')}")
        metadata = true_map.get(job_title, {"id": None, "soc_true": None})
        return {
            "job_title": job_title,
            "id": metadata["id"],
            "soc_true": metadata["soc_true"],
            "response": result
        }
    except Exception as e:
        print(f"❌ {job_title} failed: {e}")
        return {"job_title": job_title, "error": str(e)}

async def process_batch(batch: List[str]):
    async with httpx.AsyncClient() as client:
        tasks = [classify(client, title) for title in batch]
        return await asyncio.gather(*tasks)

async def main():
    all_results = []
    total_batches = math.ceil(len(JOB_TITLES) / BATCH_SIZE)

    for i in range(total_batches):
        batch = JOB_TITLES[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        print(f"\nProcessing batch {i + 1}/{total_batches} ({len(batch)} items)")
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)

        await asyncio.sleep(5)

    # Save all results
    with open("./test_results/verian_classify_results_1.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n Finished. {len(all_results)} job titles processed and saved to test_results/verian_classify_results_1.json")

if __name__ == "__main__":
    asyncio.run(main())