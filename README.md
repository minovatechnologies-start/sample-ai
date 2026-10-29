# Minova Technologies Coding AI

A small, local-first scaffold for collecting and cleaning coding examples, preparing a leakage-aware JSONL dataset, fine-tuning a reviewed open-weight model with LoRA, evaluating held-out examples, and serving inference over an authenticated FastAPI endpoint.

This is starter infrastructure, not a trained model. It does not download datasets, pull Docker images, or start training automatically. The pilot has one synthetic example per language plus one Python unit-test-generation example; it is for exercising the pipeline, not for meaningful model evaluation.

## Supported languages and task types

Languages: Python, JavaScript, TypeScript, C, C++, Java, HTML, CSS, SQL, Go, and Rust.

Every dataset example must declare one task type: `code_generation`, `debugging`, `code_explanation`, or `unit_test_generation`.

## Windows setup

Use Python 3.10 or newer. From PowerShell in the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Edit `.env` before training or serving. Review the exact model card and confirm that its license and access conditions permit your intended use. `Qwen/Qwen2.5-Coder-1.5B-Instruct` is a suggested small starter candidate; verify its current terms rather than relying on this README. Set `MODEL_LICENSE` to the verified SPDX ID and set `MODEL_LICENSE_REVIEWED=true` only after that review. The training/evaluation/API code enforces this acknowledgement and the built-in permissive-license allowlist before model loading. For gated models, authenticate with Hugging Face manually or configure `HF_TOKEN` in the untracked `.env`. The scripts download weights only when you explicitly run training, evaluation, or the API. Model downloads can still be several gigabytes.

The `tree-sitter-language-pack` syntax parsers are used to check every supported language during dataset preparation. Parser errors and unsupported or ambiguous language detection are rejected rather than silently accepted.

## 1. Prepare data

Inputs can be local `.jsonl`, `.json`, or `.csv` files. Records may use normalized names (`prompt`, `completion`) or common aliases (`instruction`/`input`, `output`/`answer`). Each record must include:

```json
{
  "id": "example-001",
  "task_type": "code_generation",
  "language": "python",
  "prompt": "Write a function that adds two integers.",
  "completion": "def add(a, b):\n    return a + b",
  "provenance": {
    "source_name": "Your approved source",
    "source_uri": "https://example.invalid/dataset",
    "repository": "optional repository/group identifier",
    "revision": "optional commit or version",
    "license": "MIT"
  },
  "tests": {
    "stdin": "",
    "expected_stdout": "3"
  }
}
```

The pipeline preserves provenance, rejects records with missing provenance or licenses outside the allowlist, detects common secrets and email addresses, checks language/syntax, removes exact duplicates, and writes a deterministic split grouped by repository (or source URI if repository is absent). It refuses to split fewer than two provenance groups. Review `rejected.jsonl` and `report.json`; rejected records store only an ID and reason, not the source content.

```powershell
python -m minova_ai.data_pipeline --input data\raw --output-dir data\processed --validation-ratio 0.2
```

Only permissive/CC0 identifiers are allowed by default. This is a policy starting point, not a legal determination. Override only after checking the dataset's actual terms and attribution/share-alike obligations:

```powershell
python -m minova_ai.data_pipeline --input data\raw --output-dir data\processed --allowed-licenses "CC0-1.0,MIT,Apache-2.0"
```

Never assume a dataset's license from its host, repository visibility, or a model card for a different release. Keep source URLs and revisions accurate. Avoid personal, confidential, and secret-bearing data; this filter is only a safety net, not a guarantee.

The included `data\examples\pilot.jsonl` contains small Minova-authored synthetic examples marked `CC0-1.0` and unique synthetic provenance references. Use it to check normalization only:

```powershell
python -m minova_ai.data_pipeline --input data\examples\pilot.jsonl --output-dir data\processed
```

## 2. Prepare and train

When your approved dataset is ready in `data\raw`, run the Python-only pipeline below. It validates and splits the raw records first, then starts LoRA training using those generated splits. If data preparation fails, training does not start.

```powershell
python -m minova_ai.pipeline
```

You can also use `minova-pipeline` after installing the project with `pip install -e .`. Set `RAW_DATA_DIR` and `PROCESSED_DATA_DIR` in `.env` to use different folders. The pipeline does not watch the folder or train automatically; run it explicitly when the dataset is ready.

The pipeline accepts JSON Lines even when the file extension is `.json`, maps `generate`/`fix`/`explain` to the supported task types, and can filter to the configured `TRAIN_LANGUAGES` (Python by default). For records without provenance, it uses the explicit `DATASET_SOURCE_NAME`, `DATASET_SOURCE_URI`, and `DATASET_LICENSE` values in `.env`; add proprietary dataset licenses to the configured allowlist only after confirming usage rights. Code blocks in step-by-step answers are syntax-checked without stripping the original response used for training.

The synthetic pilot is not placed in `data\raw` and is not a useful training set. When all records have a single source and no repository groups, the pipeline uses a deterministic record-level validation split; this cannot prevent semantic overlap across related examples. The default local model is Qwen2.5-Coder-0.5B-Instruct; larger models require suitable hardware. No model download or training occurs until you explicitly run the command.

## 3. Fine-tune separately (manual; GPU recommended)

Fine-tuning explicitly downloads the configured base model and can consume substantial storage and compute. Review the model's license, gated-access conditions, and hardware requirements first. Windows supports the scaffold's standard LoRA path; QLoRA is deliberately restricted to a supported Linux/CUDA/bitsandbytes environment. Configure a validated `LORA_TARGET_MODULES` list for the selected architecture.

Install the PyTorch build that matches your CPU/CUDA setup before installing the remaining dependencies if the default package index does not provide the desired build.

```powershell
python -m minova_ai.train
```

The training code masks prompt tokens from the loss and trains on assistant completions. Settings are in `.env`. The sample set is not suitable for training a useful model. QLoRA requires a compatible `bitsandbytes` installation and CUDA GPU; the project does not install it automatically.

## 4. Evaluate the held-out split

The evaluator records exact-match, syntax/execution, and stdin/stdout test pass metrics separately by language. Generated unit-test suites are syntax-checked only; running them requires a language-specific test harness and target project fixture, which this starter does not yet provide. Programs are run only in Docker with no network, read-only root filesystem, dropped Linux capabilities, an unprivileged user, process/memory/CPU limits, and a time limit. Images are configured in `src\minova_ai\sandbox.py` or overridden with `MINOVA_SANDBOX_IMAGE_<LANGUAGE>`.

Install Docker Desktop and manually pull the required runtime images before evaluation (the runner uses `--pull=never`). Verify Docker's isolation settings for your environment. Never enable a host-execution fallback. HTML, CSS, and SQL receive syntax-only parser checks; TypeScript executes in Node's type-stripping mode, not a full `tsc` type-check. SQL queries are not executed because the scaffold has no application schema. The I/O test contract for executable languages is `stdin` plus exact `expected_stdout`. Containers are defense in depth, not a guarantee against container-runtime/kernel escapes; use a hardened VM-based runtime for hostile public submissions.

```powershell
python -m minova_ai.evaluate --input data\processed\validation.jsonl --output monitoring\evaluation.json
```

Evaluation requires the fine-tuned adapter, so it is not available until training has been run. A missing Docker installation/image produces unavailable/failed execution results; inspect the report rather than treating those as passes.

For a single source file without the model:

```powershell
python -m minova_ai.cli --language python --source path\to\program.py --tests path\to\tests.json
```

## 5. Serve the model

Set a fresh high-entropy `API_KEY` in `.env` (do not commit `.env`). The API refuses to start without one. When the adapter directory is available, the model loads on startup. If `ADAPTER_PATH` is missing or unavailable, the service starts in a degraded state: `/health` reports `degraded` and `/generate` returns HTTP 503 until the adapter is made available and the service is restarted. This API is a development scaffold; put it behind TLS, a rate limiter, and an authenticated gateway before exposure outside a trusted network.

```powershell
uvicorn minova_ai.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` in your browser for the built-in chat page. Enter the `API_KEY` from `.env`, choose a language and task, and send a prompt to test generation. The page uses the same-origin API and does not save the key. `GET /health` reports readiness; `POST /generate` requires an `Authorization` header using the `Bearer` scheme and the configured API key. Inference does not execute generated code. Use the isolated runner only when a separate evaluation step needs to execute it.

## Layout

- `src\minova_ai\data_pipeline.py`: ingestion, normalization, filters, validation, deduplication, provenance-aware split.
- `src\minova_ai\pipeline.py`: one-command raw-data preparation followed by training.
- `src\minova_ai\train.py`: explicit Transformers + PEFT LoRA/QLoRA supervised fine-tuning.
- `src\minova_ai\evaluate.py`: held-out generation and per-language metrics.
- `src\minova_ai\sandbox.py`: Docker-only execution/checks, no host fallback.
- `src\minova_ai\api.py`: bearer-authenticated FastAPI generation endpoint.
- `data\raw\`: place approved training source files here.
- `data\examples\pilot.jsonl`: tiny synthetic data for pipeline smoke checks only.
- `tests\`: offline unit tests for dataset handling.

## Run tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

No training, deployment, or GPU run is claimed by this scaffold.
