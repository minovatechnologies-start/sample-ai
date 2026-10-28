from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from minova_ai.schema import LANGUAGES
from tree_sitter_language_pack import get_parser

load_dotenv()

MAX_SOURCE_CHARS = 250_000
MAX_OUTPUT_CHARS = 20_000
IMAGE_DEFAULTS = {
    "python": "python:3.12-slim",
    "javascript": "node:22-alpine",
    "typescript": "node:22-alpine",
    "c": "gcc:14",
    "cpp": "gcc:14",
    "java": "eclipse-temurin:21",
    "go": "golang:1.24-alpine",
    "rust": "rust:1.86-slim",
}
SOURCE_NAMES = {
    "python": "main.py",
    "javascript": "main.js",
    "typescript": "main.ts",
    "c": "main.c",
    "cpp": "main.cpp",
    "java": "Main.java",
    "go": "main.go",
    "rust": "main.rs",
}
RUN_COMMANDS = {
    "python": ["python", "-I", "/work/main.py"],
    "javascript": ["node", "/work/main.js"],
    "typescript": ["node", "--experimental-strip-types", "/work/main.ts"],
    "c": ["sh", "-c", "gcc -std=c17 -O2 -o /tmp/main /work/main.c && /tmp/main"],
    "cpp": ["sh", "-c", "g++ -std=c++17 -O2 -o /tmp/main /work/main.cpp && /tmp/main"],
    "java": ["sh", "-c", "mkdir -p /tmp/classes && javac -d /tmp/classes /work/Main.java && java -cp /tmp/classes Main"],
    "go": ["sh", "-c", "cd /tmp && GO111MODULE=off GOPROXY=off go run /work/main.go"],
    "rust": ["sh", "-c", "rustc --edition=2021 /work/main.rs -o /tmp/main && /tmp/main"],
}


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    passed: bool | None
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None


def _static_syntax_check(language: str, source: str) -> ExecutionResult:
    try:
        if language == "python":
            ast.parse(source)
        else:
            tree = get_parser(language).parse(source.encode("utf-8"))
            if tree.root_node.has_error:
                raise ValueError(f"syntax parser reported errors for {language}")
    except (SyntaxError, ValueError) as exc:
        return ExecutionResult("failed", False, stderr=str(exc))
    return ExecutionResult("passed", True)


def _docker_command(
    language: str,
    temp_dir: Path,
    image: str,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs",
        "/tmp:rw,nosuid,size=64m",
        "--mount",
        f"type=bind,src={temp_dir.resolve()},dst=/work,readonly",
        "-e",
        "HOME=/tmp",
        "-i",
        image,
    ]
    return command + RUN_COMMANDS[language]


def validate_or_run(
    language: str,
    source: str,
    tests: dict[str, str] | None = None,
    timeout_seconds: int = 5,
) -> ExecutionResult:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError("source exceeds the configured size limit")
    if timeout_seconds < 1 or timeout_seconds > 30:
        raise ValueError("timeout_seconds must be between 1 and 30")
    if tests is not None:
        if not isinstance(tests, dict) or set(tests) != {"stdin", "expected_stdout"}:
            raise ValueError("tests must contain stdin and expected_stdout strings")
        if any(not isinstance(value, str) for value in tests.values()):
            raise ValueError("test stdin and expected_stdout values must be strings")
        if any(len(value) > MAX_SOURCE_CHARS for value in tests.values()):
            raise ValueError("test input or expected output exceeds the configured size limit")
    if tests is not None and language in {"html", "css", "sql"}:
        raise ValueError("HTML, CSS, and SQL use syntax-only validation; I/O tests are unsupported")
    if language in {"html", "css", "sql"}:
        return _static_syntax_check(language, source)
    if shutil.which("docker") is None:
        return ExecutionResult("unavailable", None, stderr="Docker CLI was not found.")

    image = os.getenv(f"MINOVA_SANDBOX_IMAGE_{language.upper()}", IMAGE_DEFAULTS[language])
    with tempfile.TemporaryDirectory(prefix="minova-sandbox-") as directory:
        temp_dir = Path(directory)
        source_name = SOURCE_NAMES[language]
        (temp_dir / source_name).write_text(source, encoding="utf-8")
        cid_file = temp_dir / "container.id"
        cmd = _docker_command(language, temp_dir, image)
        cmd[2:2] = ["--cidfile", str(cid_file)]
        try:
            completed = subprocess.run(
                cmd,
                input=(tests or {}).get("stdin", ""),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            container_id = cid_file.read_text(encoding="ascii").strip() if cid_file.is_file() else ""
            cleanup_error = ""
            if re.fullmatch(r"[a-f0-9]{12,64}", container_id):
                try:
                    stopped = subprocess.run(
                        ["docker", "kill", container_id],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                    if stopped.returncode != 0:
                        cleanup_error = " Docker container cleanup failed; inspect Docker."
                except subprocess.TimeoutExpired:
                    cleanup_error = " Docker container cleanup failed; inspect Docker."
            return ExecutionResult(
                "timeout",
                False,
                stderr="Execution exceeded its time limit." + cleanup_error,
            )
        except OSError as exc:
            return ExecutionResult("unavailable", None, stderr=f"Could not start Docker: {exc}")

    stdout = completed.stdout[:MAX_OUTPUT_CHARS]
    stderr = completed.stderr[:MAX_OUTPUT_CHARS]
    passed = completed.returncode == 0
    if passed and tests is not None and stdout.rstrip("\r\n") != tests["expected_stdout"].rstrip("\r\n"):
        passed = False
        stderr = (stderr + "\nExpected stdout did not match.").strip()
    return ExecutionResult("passed" if passed else "failed", passed, stdout, stderr, completed.returncode)


def validate_source(language: str, source: str, timeout_seconds: int = 5) -> ExecutionResult:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError("source exceeds the configured size limit")
    if timeout_seconds < 1 or timeout_seconds > 30:
        raise ValueError("timeout_seconds must be between 1 and 30")
    return _static_syntax_check(language, source)
