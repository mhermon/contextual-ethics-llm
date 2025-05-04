import os
import json
import pandas as pd
from openai import OpenAI
from google import genai
import anthropic
import argparse

from tqdm import tqdm
import time

GOOGLE_RPM_MAX = 20

def set_provider(provider: str):
    if provider == "google":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = "gemini-2.0-flash-lite"
    elif provider == "openai":
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = "gpt-4.1-nano"
    elif provider == "deepseek":
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        model = "deepseek-chat"
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = "claude-3-7-sonnet-20250219"
    else:
        raise ValueError("Unsupported provider. Use 'google', 'openai', 'deepseek', or 'anthropic'.")
    return client, model

def main(dataset_path: str, batch_path: str, mode: str = "batch", results_output: str = None, provider: str = "openai"):
    # 2. Load your dataset (must have a 'prompt' column)
    df = pd.read_csv(dataset_path)
    client, model = set_provider(provider)

    if mode == "batch":
        if provider != "openai":
            raise NotImplementedError("Batch mode is only supported for OpenAI")
        # 3. Build batch tasks in JSONL format
        tasks = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sync processing"):
            tasks.append({
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": row["prompt"],
                }
            })
        # 3a. Write tasks to a .jsonl file
        with open(batch_path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")

        # 4. Upload the batch file
        batch_file = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )

        # 5. Create the batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )
        print(f"Batch job {batch_job.id} submitted, status: {batch_job.status}")
    else:
        if not results_output:
            raise ValueError("results_output path is required for sync mode")
        if provider == "google":
            rate_limit_interval = 60.0 / GOOGLE_RPM_MAX  # seconds per request
            last_request_time = time.time() - rate_limit_interval
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sync processing"):
            if provider == "openai":
                resp = client.responses.create(model=model, input=row["prompt"])
                text = resp.output[0].content[0].text
            elif provider == "google":
                now = time.time()
                elapsed = now - last_request_time
                if elapsed < rate_limit_interval:
                    time.sleep(rate_limit_interval - elapsed)
                last_request_time = time.time()
                resp = client.models.generate_content(model=model, contents=[row["prompt"]])
                text = resp.text
            elif provider == "deepseek":
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    stream=False
                )
                text = resp.choices[0].message.content
            elif provider == "anthropic":
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": row["prompt"]}],
                )
                text = message.content
            results.append({
                "id": row["id"],
                "response": text
            })
        with open(results_output, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        print(f"Synchronous processing complete. Results saved to {results_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and submit an OpenAI batch job")
    parser.add_argument("--dataset", "-d", required=True, help="Path to CSV dataset with a 'prompt' column")
    parser.add_argument("--batch-output", "-o", required=False, help="Path for output JSONL batch requests file")
    parser.add_argument(
        "--mode", "-m",
        choices=["batch", "sync"],
        default="batch",
        help="Processing mode: 'batch' (default) or 'sync'"
    )
    parser.add_argument(
        "--results-output", "-r",
        help="Path to save results (required for 'sync' mode)"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["google", "openai", "deepseek", "anthropic"],
        default="openai",
        help="Provider to use: 'google', 'openai', 'deepseek', or 'anthropic' (default: 'openai')"
    )
    args = parser.parse_args()
    if args.mode == "batch" and not args.batch_output:
        parser.error("--batch-output is required when mode 'batch'")
    if args.mode == "sync" and not args.results_output:
        parser.error("--results-output is required when mode 'sync'")
    if args.mode == "batch" and args.provider != "openai":
        parser.error("Batch mode only supported with --provider openai")
    main(args.dataset, args.batch_output, args.mode, args.results_output, args.provider)