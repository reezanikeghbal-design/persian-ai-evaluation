# Persian AI Evaluation & Benchmark Platform

A lightweight, local-first platform for evaluating LLM and Retrieval-Augmented Generation (RAG) workflows on Persian-language questions.

This project demonstrates a practical evaluation pipeline covering Persian text normalization, dataset validation, document chunking, vector retrieval, LLM provider abstraction, evaluation metrics, experiment persistence, API development, analytics, automated testing, and basic MLOps integration.

> **Project scope:** This is an MVP research and evaluation tool, not a scientifically validated benchmark suite. Several evaluation metrics are lexical heuristics and should not be interpreted as proof of factual correctness or semantic quality.

## Problem Statement

Evaluating Persian-language LLM applications introduces challenges involving Persian and Arabic character variants, half-spaces, retrieval quality, answer grounding, and reproducible experiments.

This project provides a small, independent Persian golden dataset and a repeatable local benchmark workflow to inspect retrieval and generation behavior.

## Architecture

```text
       Golden JSONL Dataset
       Persian Sample Documents
                  |
                  v
          Dataset Validation
                  |
                  v
        Persian Text Normalization
                  |
                  v
            Text Chunking
                  |
                  v
         Embedding Generation
                  |
                  v
           Vector Retrieval
                  |
                  v
            LLM Provider
          /       |       \
        Mock   OpenAI    Gemini
                  |
                  v
         Evaluation Metrics
                  |
                  v
         Experiment Persistence
             /          \
          SQLite      CSV / JSON
             |
       Optional MLflow
             |
        +----+----+
        |         |
      FastAPI  Streamlit
```

All components are part of a single Python project.

The default Hashing Embedder and Mock Provider allow the core workflow to run locally without paid API access or model downloads.

For semantic retrieval, Sentence Transformers can be installed as an optional dependency. FAISS is used when available; otherwise, the retriever falls back to NumPy similarity search.

## Technology Stack

- **Language:** Python 3.11+
- **API:** FastAPI, Pydantic
- **Database:** SQLAlchemy, SQLite
- **Data Processing:** NumPy, Pandas
- **Retrieval:** Hashing Embedder, Sentence Transformers (optional), FAISS (optional)
- **LLM Providers:** Mock, OpenAI-compatible API, Google Gemini
- **Dashboard:** Streamlit, Plotly
- **Experiment Tracking:** MLflow (optional)
- **Testing:** Pytest
- **Deployment:** Docker, Docker Compose
- **CI:** GitHub Actions

## Features

- Persian text normalization
- JSONL dataset validation
- Persian document chunking
- Configurable chunk size and overlap
- Hashing-based lexical retrieval
- Optional multilingual semantic embeddings
- Optional FAISS vector indexing
- Pluggable LLM provider architecture
- Mock provider for offline testing
- Retrieval and generation evaluation
- Experiment persistence using SQLite
- CSV and JSON benchmark exports
- FastAPI endpoints for benchmark execution and result inspection
- Streamlit analytics dashboard
- Optional MLflow experiment tracking
- Automated tests and GitHub Actions workflow

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/persian-ai-evaluation.git
cd persian-ai-evaluation
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Git Bash**

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file from the example:

**Linux / macOS / Git Bash**

```bash
cp .env.example .env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

The default configuration uses the Mock provider and local SQLite database.

Never commit your `.env` file or API credentials.

### 5. Validate the dataset

```bash
python scripts/validate_dataset.py
```

### 6. Run automated tests

```bash
pytest -q
```

### 7. Run a local benchmark

```bash
python scripts/run_benchmark.py --provider mock --embedding-model hashing
```

The benchmark runs with the offline Mock provider and deterministic Hashing Embedder.

Results are saved in:

- SQLite: `data/evaluation.db`
- CSV and JSON: `artifacts/`

### 8. Start the API

```bash
uvicorn app.api.main:app --reload
```

API base URL:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 9. Start the dashboard

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

Dashboard URL:

```text
http://localhost:8501
```

## Docker

Docker and Docker Compose must be installed locally.

First, create the environment file:

```bash
cp .env.example .env
```

Then start both services:

```bash
docker compose up --build
```

Available services:

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

The Compose configuration mounts the local `data/` and `artifacts/` directories into the containers so that benchmark outputs persist locally.

To stop the services:

```bash
docker compose down
```

To initialize the database manually:

```bash
python scripts/init_db.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/datasets` | List available datasets |
| GET | `/models` | List provider/model options |
| POST | `/experiments/run` | Execute a benchmark |
| GET | `/experiments` | List saved experiments |
| GET | `/experiments/{experiment_id}` | Retrieve experiment summary |
| GET | `/results/{experiment_id}` | Retrieve per-question results |
| GET | `/metrics/summary` | Retrieve aggregate metrics |

### Example Benchmark Request

```bash
curl -X POST http://127.0.0.1:8000/experiments/run \
  -H "Content-Type: application/json" \
  -d '{"provider":"mock","embedding_model":"hashing","top_k":3}'
```

The API currently executes benchmarks synchronously. Benchmark execution errors are returned as HTTP 400 responses. The API is intended for local experimentation and is not hardened for public production deployment.

## LLM Provider Architecture

The project uses a provider abstraction to make it possible to switch between different LLM backends.

The provider contract is defined in:

```text
app/providers/base.py
```

A provider exposes:

- `name`
- `model`
- `generate(question, context, system_prompt)`

The generation result includes:

- `answer`
- `model`
- `input_tokens`
- `output_tokens`
- `latency_ms`

Provider registration is handled in:

```text
app/providers/factory.py
```

### Supported Providers

| Provider | Description |
|---|---|
| `mock` | Offline provider that returns a context extract for pipeline testing |
| `openai-compatible` | OpenAI-compatible chat completions API |
| `gemini` | Google Gemini API |

The OpenAI-compatible and Gemini integrations require their respective optional dependencies and API credentials.

Store credentials in environment variables. Never commit API keys.

## Retrieval and Embeddings

The retrieval pipeline supports two embedding approaches.

### Hashing Embedder

The default fallback embedder uses deterministic token hashing to produce fixed-size vectors.

Characteristics:

- No external model download
- No paid API requirement
- Deterministic output
- Lightweight local execution
- Lexical rather than semantic representation

This method is useful as a baseline for pipeline testing, but it should not be treated as a semantic embedding model.

### Sentence Transformers

The optional Sentence Transformer integration supports multilingual embedding models.

The default model configuration is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Install optional dependencies:

```bash
pip install -r requirements-optional.txt
```

Configure the embedding model in `.env` when using semantic retrieval.

### Vector Search

The retriever uses FAISS when the optional dependency is available.

Otherwise, it performs similarity search using NumPy.

## Dataset

The included golden dataset is located at:

```text
datasets/golden_dataset.jsonl
```

It contains 51 Persian-language questions across:

- General Knowledge
- Technology
- Science
- Finance
- Mathematics
- Persian Language
- Document-based Questions

Each JSONL record contains the following required fields:

- `id`
- `category`
- `question`
- `reference_answer`
- `context`
- `difficulty`

The optional `relevant_chunk_id` field identifies the relevant source document for retrieval evaluation.

Example format:

```json
{
  "id": "example_001",
  "category": "technology",
  "question": "نمونه سؤال فارسی",
  "reference_answer": "پاسخ مرجع",
  "context": "متن مرجع برای پاسخ‌گویی",
  "difficulty": "easy",
  "relevant_chunk_id": "technology.txt:0"
}
```

### Adding a Dataset

Add one UTF-8 JSON object per line to a JSONL file.

Validate a dataset before benchmarking:

```bash
python scripts/validate_dataset.py path/to/file.jsonl
```

The included dataset is a small independent evaluation fixture. It should be expanded and independently reviewed before being used for research claims.

## Evaluation Metrics and Limitations

The benchmark calculates retrieval, generation, grounding, latency, and estimated cost metrics.

| Metric | Implementation | Type | Limitation |
|---|---|---|---|
| Recall@K | Relevant source chunks retrieved in top-K / all labeled relevant chunks | Exact given correct labels | Depends on dataset relevance labels; questions without labels receive zero |
| MRR | Reciprocal rank of the first relevant retrieved chunk | Exact given correct labels | Depends on dataset relevance labels |
| Answer Relevancy | Question/answer token overlap after Persian text normalization | Heuristic | Does not understand semantic equivalence |
| Faithfulness | Share of answer tokens found in retrieved context | Heuristic | Cannot verify factual claims or semantic entailment |
| Hallucination Rate | Fraction of answers with a grounding score below 0.5 | Heuristic | Weak lexical warning, not a reliable hallucination detector |
| Latency | Provider-reported generation latency | Provider-reported | Does not necessarily represent end-to-end pipeline latency |
| Token Usage | Provider-reported token metadata | Provider-reported | May be unavailable; stored as null |
| Estimated Cost | Configured model pricing multiplied by reported token counts | Estimate | Requires known model pricing and provider-reported token counts |

Recall@K and MRR are ranking metrics whose reliability depends on the completeness and correctness of dataset relevance labels.

Answer Relevancy, Faithfulness, and Hallucination Rate are lightweight lexical heuristics. They are useful for pipeline inspection and regression checks, but they are not substitutes for human evaluation or validated semantic evaluation.

LLM-as-a-judge and Ragas are intentionally not required for the MVP. They can be added later with a documented judge prompt, calibration set, and reproducibility controls.

## Experiment Tracking and Persistence

Each benchmark run is persisted in SQLite:

```text
data/evaluation.db
```

Per-question results are also exported as CSV and JSON files under:

```text
artifacts/
```

MLflow integration is optional.

To enable it, configure:

```env
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=./mlruns
```

Install the optional dependencies before enabling MLflow:

```bash
pip install -r requirements-optional.txt
```

SQLite remains the primary persistence layer. MLflow logging failures are handled without discarding saved SQLite experiment results.

## Example Benchmark Results

A recorded local Mock-provider experiment is included in:

```text
examples/benchmark_summary.json
examples/benchmark_results.csv
```

The recorded experiment evaluated 51 questions using the deterministic Hashing Embedder and Mock Provider.

These results demonstrate that the benchmark pipeline executes and persists experiment outputs. They are not evidence of real LLM quality, semantic retrieval performance, or factual accuracy.

Run the benchmark locally to generate fresh results under `artifacts/`.

## Screenshots

Screenshots of the dashboard and benchmark results can be added here.

## Limitations

- The default Hashing Embedder is a deterministic lexical baseline, not a semantic embedding model.
- FAISS is used when installed; otherwise, NumPy similarity search is used.
- The Mock Provider is intended for pipeline testing, not real model-quality evaluation.
- Faithfulness and Answer Relevancy are simple lexical heuristics.
- Hallucination Rate is a weak grounding-based warning, not a factuality detector.
- Recall@K and MRR depend on the quality and completeness of relevance annotations.
- Provider-reported latency does not necessarily represent total pipeline execution time.
- Cost estimation requires provider-reported token counts and configured model pricing.
- The included dataset is small and requires further independent review.
- The API is designed for local experimentation and does not include production-grade authentication or security hardening.
- LLM-as-a-judge and Ragas are outside the current MVP scope.

## Future Improvements

- Expand and version the Persian golden dataset.
- Add expert-reviewed retrieval relevance annotations.
- Introduce calibrated LLM-as-a-judge evaluation.
- Add optional Ragas integration.
- Compare multilingual embedding models.
- Add retrieval ablation experiments.
- Improve benchmark reporting and experiment comparisons.
- Add configurable per-model pricing and token-cost reports.
- Improve dashboard filtering and experiment comparison.
- Add production-oriented API security and deployment controls.

## Repository Structure

```text
persian-ai-evaluation/
│
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── benchmark/        # Benchmark execution
│   ├── core/             # Configuration, schemas, logging, text utilities
│   ├── database/         # SQLAlchemy models and database setup
│   ├── evaluation/       # Dataset validation and evaluation metrics
│   ├── providers/        # Mock, OpenAI-compatible, Gemini providers
│   └── rag/              # Chunking, embeddings, retrieval, RAG engine
│
├── config/               # Metric baseline and model pricing
├── dashboard/            # Streamlit analytics dashboard
├── datasets/              # Golden dataset and sample documents
├── examples/              # Example benchmark outputs
├── scripts/               # Dataset and benchmark CLI utilities
├── tests/                 # Automated tests
├── .github/workflows/     # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).