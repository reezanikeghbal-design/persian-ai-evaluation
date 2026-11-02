import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.dataset import load_jsonl, validate_dataset

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/golden_dataset.jsonl")
records = load_jsonl(path)
errors = validate_dataset(records)
if errors:
    print("Dataset validation failed:")
    print("\n".join(errors))
    raise SystemExit(1)
print(f"Dataset valid: {len(records)} records")
