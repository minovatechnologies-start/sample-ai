from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from minova_ai.policy import require_model_license_review

# Load optional local-development settings without overriding deployment environment variables.
load_dotenv()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12_000)
    language: Literal[
        "python", "javascript", "typescript", "c", "cpp", "java",
        "html", "css", "sql", "go", "rust",
    ]
    task_type: Literal[
        "code_generation", "debugging", "code_explanation", "unit_test_generation",
    ] = "code_generation"
    max_new_tokens: int = Field(default=256, ge=1, le=1024)


class GenerateResponse(BaseModel):
    completion: str
    model: str


def _load_model() -> tuple[Any, Any, str]:
    model_id = require_model_license_review()
    adapter_path = os.getenv("ADAPTER_PATH", "").strip()
    if not adapter_path or not os.path.isdir(adapter_path):
        raise FileNotFoundError("ADAPTER_PATH must point to a trained PEFT adapter directory.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_options = {"device_map": "auto"} if torch.cuda.is_available() else {}
    base = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer, model_id


def _load_api_key() -> str:
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "API_KEY is missing. Set it in the environment or in a local .env file "
            "before starting the inference server."
        )
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = _load_api_key()
    app.state.model, app.state.tokenizer, app.state.model_id = _load_model()
    yield
    app.state.model = None
    app.state.tokenizer = None


app = FastAPI(title="Minova Coding AI", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def chatbot() -> FileResponse:
    return FileResponse(Path(__file__).with_name("static") / "chat.html")


def _authenticate(request: Request, authorization: str | None = Header(default=None)) -> None:
    expected = request.app.state.api_key
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "model": getattr(request.app.state, "model_id", "not_loaded")}


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(_authenticate)])
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    prompt = (
        f"Task: {payload.task_type}\nLanguage: {payload.language}\n"
        f"Instruction:\n{payload.prompt}\nResponse:\n"
    )
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=payload.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return GenerateResponse(completion=completion, model=request.app.state.model_id)
