import json
from pathlib import Path
from typing import Any

REQUIRED = {"id", "category", "question", "reference_answer", "context", "difficulty"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def validate_dataset(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"Record {index}: must be an object")
            continue
        missing = REQUIRED - record.keys()
        if missing:
            errors.append(f"Record {index}: missing fields {sorted(missing)}")
            continue
        record_id = str(record["id"])
        if record_id in seen:
            errors.append(f"Record {index}: duplicate id {record_id}")
        seen.add(record_id)
        if not str(record["question"]).strip():
            errors.append(f"Record {index}: empty question")
        if not str(record["reference_answer"]).strip():
            errors.append(f"Record {index}: empty reference_answer")
        if not str(record["category"]).strip() or not str(record["difficulty"]).strip():
            errors.append(f"Record {index}: empty category or difficulty")
    return errors
