# Persian AI Evaluation & Benchmark Platform

![CI](https://github.com/reezanikeghbal-design/persian-ai-evaluation/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/github/license/reezanikeghbal-design/persian-ai-evaluation)

A local-first evaluation and benchmarking platform for **Persian-language Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems**.

The project provides a reproducible workflow for inspecting dataset quality, Persian text normalization, document chunking, embedding and retrieval behavior, LLM provider outputs, evaluation metrics, and experiment history. It combines a FastAPI backend, a Streamlit analytics dashboard, and lightweight local persistence.

> **Research scope:** This repository is an MVP engineering and experimentation platform, not a scientifically validated benchmark suite. Several answer-quality metrics are lexical heuristics; they should not be interpreted as proof of factual correctness or semantic quality.

## Why This Project?

Evaluating Persian-language AI applications involves language-specific and system-level challenges, including Persian and Arabic character variants, half-spaces, retrieval relevance, answer grounding, and experiment reproducibility.

This project provides a small Persian golden dataset and a repeatable local workflow to help developers inspect these behaviors, compare configurations, and identify areas for improvement.

## Key Features

- **Persian text processing:** Normalize Persian text and validate JSONL evaluation datasets.
- **Document preparation:** Chunk Persian documents with configurable chunk size and overlap.
- **Flexible retrieval:** Use a deterministic hashing baseline or optional multilingual Sentence Transformer embeddings.
- **Vector search:** Use FAISS when available, with a NumPy similarity-search fallback.
- **Provider abstraction:** Run the pipeline with an offline Mock provider or configured OpenAI-compatible and Gemini integrations.
- **Evaluation metrics:** Inspect retrieval ranking, lexical answer relevance, grounding heuristics, latency, token usage, and estimated cost.
- **Experiment persistence:** Store benchmark runs in SQLite and export per-question results as CSV and JSON.
- **Analytics dashboard:** Explore experiment data through Streamlit and Plotly.
- **Experiment tracking:** Integrate MLflow optionally.
- **Developer workflow:** Run a FastAPI service, automated tests, and Docker Compose deployment.

## Architecture

The platform is organized as a single Python project. A benchmark run validates the dataset, normalizes text, prepares chunks, generates embeddings, retrieves relevant content, invokes the selected LLM provider, computes metrics, and persists the results.

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
          /       |       \\
        Mock   OpenAI    Gemini
                  |
                  v
         Evaluation Metrics
                  |
                  v
         Experiment Persistence
             /          \\
          SQLite      CSV / JSON
             |
       Optional MLflow
             |
        +----+----+
        |         |
      FastAPI  Streamlit
```

The default Hashing Embedder and Mock Provider allow the core workflow to run locally without paid API access or model downloads. Semantic embeddings and FAISS indexing are optional.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11+ |
| API and validation | FastAPI, Pydantic |
| Persistence | SQLAlchemy, SQLite |
| Data processing | NumPy, Pandas |
| Embeddings | Hashing Embedder, Sentence Transformers (optional) |
| Retrieval | FAISS (optional), NumPy |
| LLM providers | Mock, OpenAI-compatible API, Google Gemini |
| Dashboard | Streamlit, Plotly |
| Experiment tracking | MLflow (optional) |
| Testing | Pytest |
| Deployment and CI | Docker, Docker Compose, GitHub Actions |

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- Docker and Docker Compose (optional, for containerized execution)

### 1. Clone the repository

```bash
git clone https://github.com/reezanikeghbal-design/persian-ai-evaluation.git
cd persian-ai-evaluation
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
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

### 4. Configure the environment

Create a local environment file:

**Linux / macOS / Git Bash**
```bash
cp .env.example .env
```

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

The default configuration uses the Mock provider and a local SQLite database. Set `API_KEY` in `.env` to a long, random secret before starting the API. Keep credentials in environment variables and never commit your `.env` file or API keys.

### 5. Validate the dataset and run tests

```bash
python scripts/validate_dataset.py
pytest -q
```

### 6. Run an offline benchmark

```bash
python scripts/run_benchmark.py --provider mock --embedding-model hashing
```

This command uses the offline Mock provider and deterministic Hashing Embedder. Results are saved to:

- SQLite database: `data/evaluation.db`
- CSV and JSON exports: `artifacts/`

### 7. Start the API

```bash
uvicorn app.api.main:app --reload
```

- API base URL: http://127.0.0.1:8000
- Interactive documentation: http://127.0.0.1:8000/docs

### 8. Start the dashboard

In another terminal, activate the virtual environment and run:

```bash
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

## Docker Deployment

Create the environment file, then build and start the services:

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |

The Compose configuration mounts local `data/` and `artifacts/` directories so benchmark outputs persist on the host.

Stop the services with:

```bash
docker compose down
```

To initialize the database manually:

```bash
python scripts/init_db.py
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/datasets` | List available datasets |
| `GET` | `/models` | List provider and model options |
| `POST` | `/experiments/run` | Execute a benchmark |
| `GET` | `/experiments` | List saved experiments |
| `GET` | `/experiments/{experiment_id}` | Retrieve an experiment summary |
| `GET` | `/results/{experiment_id}` | Retrieve per-question results |
| `GET` | `/metrics/summary` | Retrieve aggregate metrics |

### Example Benchmark Request

```bash
curl -X POST http://127.0.0.1:8000/experiments/run \\
  -H "Content-Type: application/json" \\
  -d '{"provider":"mock","embedding_model":"hashing","top_k":3}'
```

All operational API endpoints require the `X-API-Key` header. The `/health` endpoint remains public for health checks. Set `API_KEY` in `.env` before starting the service; requests without a valid key receive HTTP 401, and an unset server key disables operational endpoints with HTTP 503. Dataset paths supplied to benchmark requests must resolve to files inside the `datasets/` directory. Benchmark execution is currently synchronous. Execution errors are logged server-side and returned with a generic HTTP 400 message. The API is intended for local experimentation and is not hardened for public production deployment.

## LLM Provider Integration

A provider abstraction makes it possible to switch between supported LLM backends.

The provider contract is defined in `app/providers/base.py`, and provider registration is handled in `app/providers/factory.py`.

A provider exposes:

- `name`
- `model`
- `generate(question, context, system_prompt)`

The generation result includes the answer, model identifier, input and output token metadata, and provider-reported latency.

| Provider | Purpose |
|---|---|
| `mock` | Offline provider that returns a context extract for pipeline testing |
| `openai-compatible` | Integration with an OpenAI-compatible chat completions API |
| `gemini` | Google Gemini integration |

The OpenAI-compatible and Gemini integrations require their respective optional dependencies and API credentials. Store credentials in environment variables; never commit secrets.

## Retrieval and Embeddings

### Hashing Embedder

The default fallback embedder uses deterministic token hashing to produce fixed-size vectors.

- No external model download or paid API required
- Deterministic output and lightweight local execution
- Lexical rather than semantic representation

It is useful as a reproducible baseline for pipeline testing, but it should not be treated as a semantic embedding model.

### Sentence Transformers

Optional multilingual semantic embeddings are supported through Sentence Transformers. The default model configuration is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Install optional dependencies:

```bash
pip install -r requirements-optional.txt
```

Configure the embedding model in `.env` when using semantic retrieval.

### Vector Search

FAISS is used when the optional dependency is installed. Otherwise, the retriever falls back to NumPy similarity search.

## Persian Evaluation Dataset

The included golden dataset is located at:

```text
datasets/golden_dataset.jsonl
```

It contains **51 Persian-language questions** across these categories:

- General Knowledge
- Technology
- Science
- Finance
- Mathematics
- Persian Language
- Document-based Questions

Each JSONL record contains the required fields:

- `id`
- `category`
- `question`
- `reference_answer`
- `context`
- `difficulty`

The optional `relevant_chunk_id` field identifies a relevant source document for retrieval evaluation.

Example record:

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

### Adding and Validating Data

Add one UTF-8 JSON object per line to a JSONL file, then validate it:

```bash
python scripts/validate_dataset.py path/to/file.jsonl
```

The included dataset is a small independent evaluation fixture. Expand and independently review it before using it to support research claims.

## Evaluation Metrics

The benchmark reports retrieval, answer-quality, grounding, latency, token-usage, and estimated-cost information.

| Metric | Implementation | Interpretation and limitation |
|---|---|---|
| Recall@K | Relevant source chunks retrieved in top-K divided by all labeled relevant chunks | Exact given correct labels; depends on annotation completeness. Questions without labels receive zero. |
| MRR | Reciprocal rank of the first relevant retrieved chunk | Exact given correct labels; depends on annotation quality. |
| Answer Relevancy | Question/answer token overlap after Persian normalization | Lexical heuristic; does not capture semantic equivalence. |
| Faithfulness | Share of answer tokens found in retrieved context | Lexical heuristic; cannot verify factual claims or entailment. |
| Hallucination Rate | Fraction of answers with a grounding score below 0.5 | Weak grounding warning, not a factuality detector. |
| Latency | Provider-reported generation latency | May not represent end-to-end pipeline latency. |
| Token Usage | Provider-reported token metadata | May be unavailable and stored as null. |
| Estimated Cost | Configured model pricing multiplied by reported token counts | Requires configured pricing and provider-reported token counts. |

Recall@K and MRR are ranking metrics; their reliability depends on the completeness and correctness of relevance labels. Answer Relevancy, Faithfulness, and Hallucination Rate are lightweight lexical heuristics intended for pipeline inspection and regression checks. They are not substitutes for human review or validated semantic evaluation.

LLM-as-a-judge and Ragas are not required by this MVP. They remain potential extensions and would need documented prompts, calibration data, and reproducibility controls.

## Experiment Tracking and Results

Each benchmark run is persisted in SQLite:

```text
data/evaluation.db
```

Per-question results are exported as CSV and JSON files under:

```text
artifacts/
```

MLflow tracking is optional. To enable it, configure the following in `.env`:

```env
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=./mlruns
```

Install optional dependencies before enabling MLflow:

```bash
pip install -r requirements-optional.txt
```

SQLite remains the primary persistence layer. MLflow logging failures are handled without discarding saved SQLite experiment results.

### Included Example Results

A recorded local Mock-provider experiment is available at:

- `examples/benchmark_summary.json`
- `examples/benchmark_results.csv`

The recorded run evaluated 51 questions using the deterministic Hashing Embedder and Mock Provider. These files demonstrate benchmark execution and result persistence; they are not evidence of real LLM quality, semantic retrieval performance, or factual accuracy. Run a fresh benchmark to generate current outputs under `artifacts/`.

## Testing

Run the automated test suite with:

```bash
pytest -q
```

## Limitations

- The default Hashing Embedder is a deterministic lexical baseline, not a semantic embedding model.
- FAISS is optional; NumPy similarity search is used as a fallback.
- The Mock Provider is for pipeline testing, not real model-quality evaluation.
- Faithfulness and Answer Relevancy are lexical heuristics.
- Hallucination Rate is a weak grounding-based warning, not a factuality detector.
- Recall@K and MRR depend on the quality and completeness of relevance annotations.
- Provider-reported latency may not represent total pipeline execution time.
- Cost estimation requires provider-reported token counts and configured model pricing.
- The included dataset is small and needs further independent review.
- The API is intended for local experimentation and lacks production-grade authentication and security hardening.
- LLM-as-a-judge and Ragas are outside the current MVP scope.

## Future Improvements

- Expand and version the Persian golden dataset.
- Add expert-reviewed retrieval relevance annotations.
- Introduce calibrated LLM-as-a-judge evaluation.
- Add optional Ragas integration.
- Compare multilingual embedding models and retrieval configurations.
- Add retrieval ablation experiments.
- Improve benchmark reporting and experiment comparisons.
- Add configurable per-model pricing and token-cost reports.
- Improve dashboard filtering and experiment comparison.
- Add production-oriented API security and deployment controls.

## Repository Structure

```text
persian-ai-evaluation/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── benchmark/        # Benchmark execution
│   ├── core/             # Configuration, schemas, logging, text utilities
│   ├── database/         # SQLAlchemy models and database setup
│   ├── evaluation/       # Dataset validation and evaluation metrics
│   ├── providers/        # Mock, OpenAI-compatible, Gemini providers
│   └── rag/              # Chunking, embeddings, retrieval, RAG engine
├── config/               # Metric baseline and model pricing
├── dashboard/            # Streamlit analytics dashboard
├── datasets/             # Golden dataset and sample documents
├── examples/             # Example benchmark outputs
├── scripts/              # Dataset and benchmark CLI utilities
├── tests/                # Automated tests
├── .github/workflows/    # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
