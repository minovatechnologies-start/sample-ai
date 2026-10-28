from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tree_sitter_language_pack import get_parser

from minova_ai.schema import Example, LANGUAGES, Provenance, SCHEMA_VERSION, TASK_TYPES

MAX_INPUT_FILE_BYTES = 5_000_000
MAX_FIELD_CHARS = 250_000
DEFAULT_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "Unlicense",
}
EXTENSION_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".sql": "sql",
    ".go": "go",
    ".rs": "rust",
}
FENCE_RE = re.compile(r"```([a-zA-Z0-9+#.-]*)\s*\n(.*?)```", re.DOTALL)
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
    ),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]
LANGUAGE_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "c++": "cpp",
    "cxx": "cpp",
    "py": "python",
    "golang": "go",
    "htm": "html",
}
TASK_ALIASES = {
    "generate": "code_generation",
    "fix": "debugging",
    "explain": "code_explanation",
}


def _canonical_language(language: str) -> str:
    normalized = language.strip().lower().replace("#", "sharp")
    return LANGUAGE_ALIASES.get(normalized, normalized)


def detect_language(text: str, file_name: str = "") -> str | None:
    extension_language = EXTENSION_LANGUAGES.get(Path(file_name).suffix.lower())
    if extension_language:
        return extension_language
    fences = FENCE_RE.findall(text)
    if fences:
        labels = {_canonical_language(label) for label, _ in fences if label}
        if len(labels) == 1:
            candidate = labels.pop()
            return candidate if candidate in LANGUAGES else None
        if len(labels) > 1:
            return None

    source = text.lower()
    checks: list[tuple[str, bool]] = [
        ("html", bool(re.search(r"<(?:!doctype|html|body|div|p|section)\b", source))),
        ("sql", bool(re.search(r"\b(select|insert\s+into|create\s+table|update\s+\w+\s+set)\b", source))),
        ("rust", "fn main" in source or "println!" in source or "fn " in source and "->" in source),
        ("go", "package main" in source or "func main()" in source),
        ("java", "public static void main" in source or "system.out.println" in source),
        ("typescript", bool(re.search(r":\s*(?:string|number|boolean)\b|interface\s+\w+", source))),
        ("cpp", "std::" in source or "cout" in source or "template<" in source),
        ("css", bool(re.search(r"[^{}]+\{[^{}]*[a-z-]+\s*:\s*[^{};]+;", source))),
        ("javascript", "console.log" in source or "const " in source or "=>" in source),
        ("c", bool(re.search(r"#include\s*<stdio\.h>|int\s+main\s*\(", source))),
        ("python", "def " in source or "import " in source or "print(" in source),
    ]
    matches = [language for language, matched in checks if matched]
    return matches[0] if len(matches) == 1 else None


def _has_sensitive_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _syntax_check(language: str, code: str) -> None:
    source = code.encode("utf-8")
    if language == "python":
        ast.parse(code)
        return
    parser = get_parser(language)
    tree = parser.parse(source)
    if tree.root_node.has_error:
        raise ValueError(f"syntax parser reported errors for {language}")


def _pick(raw: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
    return default


def normalize_record(
    raw: dict[str, Any],
    file_name: str,
    row_index: int,
    source_name_default: str = "",
    source_uri_default: str = "",
    license_default: str = "",
) -> Example:
    prompt_value = _pick(raw, "prompt", "instruction", "input")
    completion_value = _pick(raw, "completion", "output", "answer", "response")
    if not isinstance(prompt_value, str) or not isinstance(completion_value, str):
        raise ValueError("prompt and completion must be strings")
    prompt = prompt_value.strip()
    completion = completion_value.strip()
    if not prompt or not completion:
        raise ValueError("prompt and completion are required")
    if len(prompt) > MAX_FIELD_CHARS or len(completion) > MAX_FIELD_CHARS:
        raise ValueError("prompt or completion exceeds the configured size limit")

    task_type = str(_pick(raw, "task_type", "task")).strip().lower()
    task_type = TASK_ALIASES.get(task_type, task_type)
    if task_type not in TASK_TYPES:
        raise ValueError("task_type must be one of the four supported task types")

    language_value = _pick(raw, "language")
    language = _canonical_language(str(language_value)) if language_value else None
    if language and language not in LANGUAGES:
        raise ValueError("unsupported language")
    if task_type == "code_explanation":
        code_in_context = "\n".join(
            body
            for text in (prompt, completion)
            for _, body in FENCE_RE.findall(text)
        )
        detected = detect_language(code_in_context, file_name) if code_in_context else None
    else:
        detected = detect_language(completion, file_name)
    if language is None:
        language = detected
    elif detected and detected != language:
        raise ValueError("declared language does not match detected language")
    if not language:
        raise ValueError("language is unknown; specify it explicitly")

    provenance_raw = raw.get("provenance", {})
    if not isinstance(provenance_raw, dict):
        raise ValueError("provenance must be an object")
    source_name = str(
        _pick(
            provenance_raw,
            "source_name",
            "dataset",
            "source",
            default=_pick(raw, "source_name", "dataset", "source", default=source_name_default),
        )
    ).strip()
    source_uri = str(
        _pick(
            provenance_raw,
            "source_uri",
            "source_url",
            "uri",
            default=_pick(raw, "source_uri", "source_url", "uri", default=source_uri_default),
        )
    ).strip()
    license_name = str(
        _pick(
            provenance_raw,
            "license",
            "license_id",
            default=_pick(raw, "license", "license_id", default=license_default),
        )
    ).strip()
    repository = str(
        _pick(provenance_raw, "repository", "repo", "repository_url", default=_pick(raw, "repository", "repo", "repository_url"))
    ).strip()
    revision = str(
        _pick(provenance_raw, "revision", "commit", "commit_sha", default=_pick(raw, "revision", "commit", "commit_sha"))
    ).strip()
    if not source_name or not source_uri or not license_name:
        raise ValueError("source_name, source_uri, and license are required")

    if _has_sensitive_data(json.dumps(raw, ensure_ascii=False)):
        raise ValueError("sensitive data detected")
    fenced_code = "\n".join(body for _, body in FENCE_RE.findall(completion))
    prompt_code = "\n".join(body for _, body in FENCE_RE.findall(prompt))
    step_code = re.search(
        r"(?is)Step\s*\d+\s*[-:]\s*(?:Write the code|Apply the fix)\s*\n"
        r"(.*?)(?:\nStep\s*\d+\b|$)",
        completion,
    )
    code_to_check = fenced_code or (step_code.group(1).strip() if step_code else "")
    code_to_check = code_to_check or (prompt_code if task_type == "code_explanation" else "")
    if task_type != "code_explanation" or code_to_check:
        try:
            _syntax_check(language, code_to_check or completion)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("syntax check failed") from exc

    tests = raw.get("tests")
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError as exc:
            raise ValueError("tests must be a valid JSON object") from exc
    if tests is not None and not isinstance(tests, dict):
        raise ValueError("tests must be an object with stdin and expected_stdout values")
    if tests is not None and set(tests) - {"stdin", "expected_stdout"}:
        raise ValueError("tests may only contain stdin and expected_stdout")
    if tests is not None and "expected_stdout" not in tests:
        raise ValueError("tests.expected_stdout is required")
    if tests is not None and any(not isinstance(value, str) for value in tests.values()):
        raise ValueError("test stdin and expected_stdout values must be strings")
    if tests is not None and any(len(value) > MAX_FIELD_CHARS for value in tests.values()):
        raise ValueError("test input or expected output exceeds the configured size limit")
    if tests is not None and language in {"html", "css", "sql"}:
        raise ValueError("I/O tests are unsupported for HTML, CSS, and SQL")
    if tests is not None and task_type not in {"code_generation", "debugging"}:
        raise ValueError("stdin/stdout tests are supported only for code_generation and debugging tasks")

    record_id = str(_pick(raw, "id", "record_id")).strip()
    if not record_id:
        record_id = hashlib.sha256(f"{file_name}:{row_index}".encode()).hexdigest()[:16]
    return Example(
        id=record_id,
        task_type=task_type,
        language=language,
        prompt=prompt,
        completion=completion,
        provenance=Provenance(source_name, source_uri, license_name, repository, revision),
        tests=tests,
    )


def _read_records(path: Path) -> Iterable[tuple[Any, str, int]]:
    files = [path] if path.is_file() else sorted(
        child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".csv"}
    )
    if not files:
        raise ValueError(f"no JSONL, JSON, or CSV input files found under {path}")
    for file_path in files:
        if file_path.stat().st_size > MAX_INPUT_FILE_BYTES:
            raise ValueError(f"input file exceeds {MAX_INPUT_FILE_BYTES} bytes: {file_path.name}")
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8-sig") as handle:
                for index, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        value = {"__invalid_json_line__": True}
                    yield value, str(file_path), index
        elif suffix == ".json":
            text = file_path.read_text(encoding="utf-8-sig")
            try:
                values = json.loads(text)
            except json.JSONDecodeError:
                values = []
                for index, line in enumerate(text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        values.append((json.loads(line), index))
                    except json.JSONDecodeError:
                        values.append(({"__invalid_json_line__": True}, index))
                for value, index in values:
                    yield value, str(file_path), index
                continue
            if isinstance(values, dict):
                values = values.get("records", [values])
            if not isinstance(values, list):
                raise ValueError(f"JSON input must be an array or object: {file_path.name}")
            for index, value in enumerate(values, start=1):
                yield value, str(file_path), index
        else:
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for index, value in enumerate(csv.DictReader(handle), start=2):
                    yield value, str(file_path), index


def _fingerprint(example: Example) -> str:
    canonical = "\n".join(
        (
            example.language,
            example.task_type,
            re.sub(r"\s+", " ", example.prompt).strip(),
            re.sub(r"\s+", " ", example.completion).strip(),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _repository_group(example: Example) -> str:
    provenance = example.provenance
    return provenance.repository or provenance.source_uri or provenance.source_name


def _split_groups(examples: list[Example], validation_ratio: float) -> tuple[list[Example], list[Example]]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[_repository_group(example)].append(example)
    if len(groups) == 1:
        ordered = sorted(
            examples,
            key=lambda example: hashlib.sha256(
                f"{_repository_group(example)}:{example.id}".encode("utf-8")
            ).hexdigest(),
        )
        validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_ratio)))
        return ordered[validation_count:], ordered[:validation_count]

    validation_groups = {
        group
        for group in groups
        if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < validation_ratio
    }
    if not validation_groups:
        validation_groups.add(min(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest()))
    if len(validation_groups) == len(groups):
        validation_groups.remove(max(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest()))
    train = [example for group, records in groups.items() if group not in validation_groups for example in records]
    validation = [example for group, records in groups.items() if group in validation_groups for example in records]
    return train, validation


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    validation_ratio: float = 0.2,
    allowed_licenses: set[str] | None = None,
    languages: set[str] | None = None,
    source_name_default: str = "",
    source_uri_default: str = "",
    license_default: str = "",
) -> dict[str, Any]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    licenses = DEFAULT_LICENSES if allowed_licenses is None else allowed_licenses
    accepted: list[Example] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    counters: Counter[str] = Counter()
    selected_languages = {_canonical_language(language) for language in languages} if languages else None

    for raw, file_name, row_index in _read_records(input_path):
        counters["read"] += 1
        record_id = (
            str(raw.get("id", raw.get("record_id", f"{Path(file_name).name}:{row_index}")))
            if isinstance(raw, dict)
            else f"{Path(file_name).name}:{row_index}"
        )
        try:
            if not isinstance(raw, dict) or raw.get("__invalid_json_line__"):
                raise ValueError("record must be a valid JSON object")
            declared_language = raw.get("language")
            if (
                selected_languages
                and declared_language
                and _canonical_language(str(declared_language)) not in selected_languages
            ):
                counters["filtered"] += 1
                continue
            example = normalize_record(
                raw,
                file_name,
                row_index,
                source_name_default,
                source_uri_default,
                license_default,
            )
            if selected_languages and example.language not in selected_languages:
                counters["filtered"] += 1
                continue
            if example.provenance.license not in licenses:
                raise ValueError("license is not in the approved allowlist")
            fingerprint = _fingerprint(example)
            if fingerprint in seen:
                counters["duplicates"] += 1
                continue
            seen.add(fingerprint)
            accepted.append(example)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, csv.Error) as exc:
            rejected.append({"id": record_id, "reason": str(exc)})
            counters["rejected"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    if len(accepted) < 2:
        raise ValueError("fewer than two valid unique examples remain; see rejected.jsonl")
    train, validation = _split_groups(accepted, validation_ratio)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    report = {
        "schema_version": SCHEMA_VERSION,
        "read": counters["read"],
        "accepted": len(accepted),
        "train": len(train),
        "validation": len(validation),
        "rejected": len(rejected),
        "filtered": counters["filtered"],
        "duplicates_removed": counters["duplicates"],
        "languages": dict(Counter(example.language for example in accepted)),
        "approved_licenses": sorted(licenses),
        "split_strategy": "provenance_group" if len({_repository_group(item) for item in accepted}) > 1 else "record_level_single_source",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_jsonl(path: Path, values: list[Example] | list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            record = value.to_dict() if isinstance(value, Example) else value
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and clean local coding datasets.")
    parser.add_argument("--input", type=Path, required=True, help="Local JSONL, JSON, CSV file or input directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--languages", default="", help="Comma-separated languages to include")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-uri", default="")
    parser.add_argument("--license", default="")
    parser.add_argument(
        "--allowed-licenses",
        default=",".join(sorted(DEFAULT_LICENSES)),
        help="Comma-separated SPDX license IDs approved for your use case",
    )
    args = parser.parse_args()
    result = prepare_dataset(
        args.input,
        args.output_dir,
        args.validation_ratio,
        {item.strip() for item in args.allowed_licenses.split(",") if item.strip()},
        {item.strip() for item in args.languages.split(",") if item.strip()} or None,
        args.source_name,
        args.source_uri,
        args.license,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
