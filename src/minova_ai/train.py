from __future__ import annotations

import os
import platform
from pathlib import Path

import torch  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv
from datasets import load_dataset  # pyright: ignore[reportMissingImports]
from minova_ai.policy import require_model_license_review
from peft import (  # pyright: ignore[reportMissingImports]
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # pyright: ignore[reportMissingImports]
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

load_dotenv()


def _format_prompt(record: dict[str, object]) -> str:
    return (
        f"Task: {record['task_type']}\n"
        f"Language: {record['language']}\n"
        f"Instruction:\n{record['prompt']}\n"
        "Response:\n"
    )


def main() -> None:
    model_id = require_model_license_review()
    max_length = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
    if max_length < 2:
        raise ValueError("MAX_SEQ_LENGTH must be at least 2")
    train_file = Path(os.getenv("TRAIN_FILE", "data/processed/train.jsonl"))
    validation_file = Path(os.getenv("VALIDATION_FILE", "data/processed/validation.jsonl"))
    output_dir = Path(os.getenv("ADAPTER_OUTPUT_DIR", "models/v1/adapter"))
    use_4bit = os.getenv("USE_4BIT", "false").lower() == "true"
    if not train_file.is_file() or not validation_file.is_file():
        raise FileNotFoundError("Prepared train.jsonl and validation.jsonl files are required.")
    if use_4bit and (platform.system() == "Windows" or not torch.cuda.is_available()):
        raise RuntimeError("QLoRA requires a supported Linux/CUDA/bitsandbytes environment; use LoRA otherwise.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer must define an EOS token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = None
    if use_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto" if use_4bit else None,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(os.getenv("LORA_R", "8")),
            lora_alpha=int(os.getenv("LORA_ALPHA", "16")),
            lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
            target_modules=os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj").split(","),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(validation_file)})

    def tokenize(record: dict[str, object]) -> dict[str, list[int]]:
        prompt = tokenizer(_format_prompt(record), add_special_tokens=True)["input_ids"]
        answer = tokenizer(str(record["completion"]), add_special_tokens=False)["input_ids"]
        answer.append(tokenizer.eos_token_id)
        answer = answer[: max_length - 1]
        prompt = prompt[: max(1, max_length - len(answer))]
        ids = prompt + answer
        labels = ([-100] * len(prompt) + answer)[: len(ids)]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(os.getenv("NUM_TRAIN_EPOCHS", "1")),
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "1")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
        logging_steps=int(os.getenv("LOGGING_STEPS", "10")),
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
