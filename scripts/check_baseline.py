import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.metrics import recall_at_k, mean_reciprocal_rank

baseline = json.loads(Path("config/metric_baseline.json").read_text(encoding="utf-8"))
observed = {"recall_at_k": recall_at_k(["relevant"], {"relevant"}, 1),
            "mrr": mean_reciprocal_rank(["relevant"], {"relevant"})}
minimum = float(baseline["minimum_allowed"])
for metric, value in observed.items():
    if value < min(minimum, float(baseline[metric])):
        raise SystemExit(f"{metric} regressed: {value} < baseline {baseline[metric]}")
print(f"Deterministic baseline passed: {observed}")
