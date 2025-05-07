from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

# Third‑party provider SDKs (import lazily when used)
from openai import OpenAI                 # OpenAI + DeepSeek (same client)
from google import genai                  # Google Gemini
import anthropic                          # Anthropic Claude

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GOOGLE_RPM_MAX: int = 10   # Gemini Flash‑lite requests per minute


# --------------------------------------------------------------------------- #
# Provider helpers
# --------------------------------------------------------------------------- #
ProviderClient = Tuple[Any, str]  # (client_handle, model_name)


def _init_openai(api_key_env: str, model: str, base_url: str | None = None) -> ProviderClient:
    client = OpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
    return client, model


def _init_google() -> ProviderClient:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client, "gemini-2.0-flash"


def _init_anthropic() -> ProviderClient:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client, "claude-3-7-sonnet-20250219"


def set_provider(provider: str) -> ProviderClient:
    """Return an SDK client handle and default model string for a provider."""
    match provider.lower():
        case "openai":
            return _init_openai("OPENAI_API_KEY", "chatgpt-4o-latest")
        case "deepseek":
            return _init_openai("DEEPSEEK_API_KEY", "deepseek-chat", base_url="https://api.deepseek.com")
        case "google":
            return _init_google()
        case "anthropic":
            return _init_anthropic()
        case _:
            raise ValueError(
                "Unsupported provider. Choose from: openai, deepseek, google, anthropic."
            )


# --------------------------------------------------------------------------- #
# Local model generation
# --------------------------------------------------------------------------- #
def generate_local_response(
    local_model_path: str,
    prompt: str,
    max_new_tokens: int,
    messages: List[Dict[str, str]] | None = None,
    verbose: bool = False,
) -> str:
    """Generate a response using a local model."""
    if "mlx" in local_model_path:
        from mlx_lm import load as mlx_load, generate as mlx_generate

        model, tokenizer = mlx_load(local_model_path)
        chat_prompt = (
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            if messages
            else prompt
        )
        return mlx_generate(
            model, tokenizer, prompt=chat_prompt, max_tokens=max_new_tokens, verbose=verbose
        )

    # Fallback to HF Transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    input_text = "".join(msg.get("content", "") for msg in messages) if messages else prompt
    return generator(input_text, max_new_tokens=max_new_tokens)[0]["generated_text"]


# --------------------------------------------------------------------------- #
# Batch‑mode helpers (OpenAI only)
# --------------------------------------------------------------------------- #
def _build_batch_tasks(df: pd.DataFrame, model: str) -> List[Dict[str, Any]]:
    """Convert each row into an OpenAI batch API task."""
    return [
        {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": model, "input": row["prompt"]},
        }
        for idx, row in df.iterrows()
    ]


def _submit_openai_batch(client: OpenAI, batch_path: Path) -> None:
    """Upload a JSONL task file then create the batch job."""
    batch_file = client.files.create(file=open(batch_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id, endpoint="/v1/responses", completion_window="24h"
    )
    print(f"Batch job {batch_job.id} submitted, status: {batch_job.status}")


# --------------------------------------------------------------------------- #
# Sync‑mode helpers
# --------------------------------------------------------------------------- #
def _respect_google_rate_limit(last_time: float) -> float:
    interval = 60.0 / GOOGLE_RPM_MAX
    elapsed = time.time() - last_time
    if elapsed < interval:
        time.sleep(interval - elapsed)
    return time.time()  # new timestamp


def _generate_response(
    provider: str,
    client: Any,
    model: str,
    prompt: str,
    local_model_path: str | None = None,
    last_request_time: float | None = None,
) -> Tuple[str, float | None]:
    """Return (response_text, updated_last_request_time)."""
    if provider == "openai":
        resp = client.responses.create(model=model, input=prompt)
        return resp.output[0].content[0].text, last_request_time

    if provider == "google":
        last_request_time = _respect_google_rate_limit(last_request_time or 0.0)
        resp = client.models.generate_content(model=model, contents=[prompt])
        return resp.text, last_request_time

    if provider == "deepseek":
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.choices[0].message.content, last_request_time

    if provider == "anthropic":
        message = client.messages.create(
            model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text, last_request_time

    if provider == "local":
        assert local_model_path, "`local_model_path` must be provided for local provider"
        text = generate_local_response(
            local_model_path,
            prompt=prompt,
            max_new_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            verbose=False,
        )
        return text, last_request_time

    raise ValueError(f"Unknown provider '{provider}'")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_batch_mode(df: pd.DataFrame, batch_path: Path, client: OpenAI, model: str) -> None:
    tasks = _build_batch_tasks(df, model)
    with open(batch_path, "w") as fp:
        for task in tasks:
            fp.write(json.dumps(task) + "\n")
    _submit_openai_batch(client, batch_path)
    print(f"Wrote {len(tasks)} tasks to {batch_path}")


def run_sync_mode(
    df: pd.DataFrame,
    results_path: Path,
    provider: str,
    client: Any,
    model: str,
    local_model_path: str | None,
) -> None:
    results: List[Dict[str, str]] = []
    last_time = 0.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sync processing"):
        text, last_time = _generate_response(
            provider, client, model, row["prompt"], local_model_path, last_time
        )
        results.append({"id": row["id"], "response": text})

    with open(results_path, "w") as fp:
        for item in results:
            fp.write(json.dumps(item) + "\n")

    print(f"Synchronous processing complete. Results saved to {results_path}")


# --------------------------------------------------------------------------- #
# CLI entry‑point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate responses via LLM providers.")
    p.add_argument("-d", "--dataset", required=True, help="CSV with a 'prompt' column")
    p.add_argument("-o", "--batch-output", help="Path for JSONL batch file (batch mode only)")
    p.add_argument(
        "-m",
        "--mode",
        choices=["batch", "sync"],
        default="batch",
        help="Processing mode (default: batch)",
    )
    p.add_argument("-r", "--results-output", help="Path for sync‑mode results JSONL")
    p.add_argument(
        "-p",
        "--provider",
        choices=["google", "openai", "deepseek", "anthropic", "local"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    p.add_argument("--local-model-path", help="Path to a local model (provider=local)")
    return p.parse_args()


def _default_results_path(dataset: Path, model_name: str) -> Path:
    return dataset.parent / f"{model_name}_output.jsonl"


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    # Provider setup
    if args.provider == "local":
        client, model_name = None, Path(args.local_model_path).name
    else:
        client, model_name = set_provider(args.provider)

    # Determine default paths
    if args.mode == "batch":
        if args.provider != "openai":
            raise ValueError("Batch mode only supported with provider 'openai'")
        if not args.batch_output:
            raise ValueError("--batch-output required for batch mode")
        batch_path = Path(args.batch_output)
    else:  # sync
        results_path = (
            Path(args.results_output)
            if args.results_output
            else _default_results_path(dataset_path, model_name)
        )
        if not args.results_output:
            print(f"No --results-output specified; using default: {results_path}")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Execute
    if args.mode == "batch":
        run_batch_mode(df, batch_path, client, model_name)
    else:
        run_sync_mode(
            df,
            results_path,  
            args.provider,
            client,
            model_name,
            args.local_model_path,
        )


if __name__ == "__main__":
    main()