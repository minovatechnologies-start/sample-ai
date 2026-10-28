from __future__ import annotations

import argparse
import json
from pathlib import Path

from minova_ai.sandbox import validate_or_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code only inside an isolated Docker container.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, help="JSON object with stdin and expected_stdout")
    args = parser.parse_args()
    tests = json.loads(args.tests.read_text(encoding="utf-8")) if args.tests else None
    result = validate_or_run(args.language, args.source.read_text(encoding="utf-8"), tests)
    print(json.dumps(result.__dict__, indent=2))
    if result.passed is False:
        raise SystemExit(1)
    if result.passed is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
