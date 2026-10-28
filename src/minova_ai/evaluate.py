from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from minova_ai.policy import require_model_license_review
from minova_ai.sandbox import validate_or_run, validate_source

load_dotenv()


def _prompt(record: dict[str, Any]) -> str:
    return (
        f"Task: {record['task_type']}\nLanguage: {record['language']}\n"
        f"Instruction:\n{record['prompt']}\nResponse:\n"
    )


def _strip_fence(value: str) -> str:
    match = re.fullmatch(r"\s*```[a-zA-Z0-9+#.-]*\s*\n(.*?)```\s*", value, flags=re.DOTALL)
    return match.group(1) if match else value


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return _strip_fence(tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))


def evaluate_file(input_file: Path, output_file: Path, max_examples: int | None = None) -> dict[str, Any]:
    model_id = require_model_license_review()
    adapter_dir = Path(os.getenv("ADAPTER_PATH", "models/v1/adapter"))
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    if not input_file.is_file():
        raise FileNotFoundError(f"Evaluation file does not exist: {input_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_options = {"device_map": "auto"} if torch.cuda.is_available() else {}
    base = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    metrics: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    evaluated = 0
    with input_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            language = str(record["language"])
            bucket = metrics[language]
            generated = _generate(model, tokenizer, _prompt(record), int(os.getenv("MAX_NEW_TOKENS", "256")))
            evaluated += 1
            bucket["examples"] += 1
            bucket["exact_match"] += int(generated.strip() == str(record["completion"]).strip())
            if record.get("tests") is not None:
                bucket["tested"] += 1
                result = validate_or_run(language, generated, record["tests"])
                bucket["test_passed"] += int(result.passed is True)
                bucket[f"check_{result.status}"] += 1
            elif record["task_type"] == "unit_test_generation":
                bucket["syntax_checked"] += 1
                result = validate_source(language, generated)
                bucket["syntax_passed"] += int(result.passed is True)
                bucket[f"check_{result.status}"] += 1
            elif record["task_type"] in {"code_generation", "debugging"}:
                bucket["syntax_checked"] += 1
                result = validate_or_run(language, generated)
                bucket["syntax_passed"] += int(result.passed is True)
                bucket[f"check_{result.status}"] += 1
            if max_examples is not None and evaluated >= max_examples:
                break

    result = {
        language: {
            **values,
            "exact_match_rate": round(values["exact_match"] / values["examples"], 4),
            "test_pass_rate": round(values["test_passed"] / values["tested"], 4) if values["tested"] else None,
            "syntax_pass_rate": (
                round(values["syntax_passed"] / values["syntax_checked"], 4)
                if values["syntax_checked"]
                else None
            ),
        }
        for language, values in sorted(metrics.items())
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps({"evaluated": evaluated, "by_language": result}, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an adapter on held-out JSONL examples.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("monitoring/evaluation.json"))
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    result = evaluate_file(args.input, args.output, args.max_examples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
