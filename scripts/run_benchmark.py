import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.benchmark.runner import BenchmarkRunner

parser = argparse.ArgumentParser(description="Run a Persian LLM/RAG benchmark")
parser.add_argument("--provider", default="mock")
parser.add_argument("--model")
parser.add_argument("--embedding-model", default="hashing")
parser.add_argument("--dataset", default="datasets/golden_dataset.jsonl")
parser.add_argument("--chunk-size", type=int, default=500)
parser.add_argument("--chunk-overlap", type=int, default=80)
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()
result = BenchmarkRunner().run(args.dataset, args.provider, args.model, args.embedding_model,
                                args.chunk_size, args.chunk_overlap, args.top_k)
result.pop("results", None)
print(json.dumps(result, ensure_ascii=False, indent=2))
