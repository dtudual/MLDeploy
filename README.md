# MLDeploy — MLOps Learning Exercise

A hands-on exercise completed as part of my **MLOps learning journey**. The goal of this repository is to understand how to **serve, containerize, deploy, and test a pretrained machine learning model**.

This is a learning exercise, not a full machine learning project or a production-ready MLOps platform.

**Live demo:** [API](https://mldeploy-7gb3.onrender.com/) · [Interactive API documentation (Swagger UI)](https://mldeploy-7gb3.onrender.com/docs)

## What I wanted to learn

Rather than training a new model, I focused on the steps that come **after training**:

1. Load and inspect an existing model.
2. Expose predictions through a REST API using FastAPI.
3. Validate inputs and handle errors.
4. Package the application and its dependencies with Docker.
5. Deploy the containerized API to Render.
6. Automate API tests with pytest and GitHub Actions.

## Overview

This exercise uses a pretrained **scikit-learn `LinearRegression`** model saved as `model.pkl`. It was trained on a small synthetic dataset following the relationship **y = 2x** and expects **one numerical feature** per prediction.

The model is loaded using `joblib` and exposed through a REST API built with FastAPI. The application is packaged with Docker, deployed on Render, and checked automatically with pytest and GitHub Actions.

### Architecture

```text
HTTP client
    |
    v
FastAPI (main.py)
    |-- Pydantic input validation
    |-- GET /health
    `-- POST /predict
             |
             v
      predictor.py
             |
             v
        model.pkl

Docker: application + model + Python dependencies
Render: public HTTPS hosting
GitHub Actions: automated pytest checks
```

## Technologies used

| Tool | Purpose |
| --- | --- |
| Python 3.12 | Application runtime |
| scikit-learn, NumPy, joblib | Model inference and data handling |
| FastAPI, Pydantic, Uvicorn | REST API, request validation, and HTTP server |
| Docker | Containerization |
| Render | Cloud deployment |
| pytest, HTTPX / TestClient | Automated API testing |
| GitHub Actions | Continuous integration (CI) |

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── ml_api_demo_main/
│   └── model.pkl
├── tests/
│   └── test_api.py
├── main.py
├── predictor.py
├── sanity_check.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Run locally

**Prerequisites:** Python 3.12 and the files in this repository, including `model.pkl`.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

On Windows PowerShell, activate the environment using `.venv\Scripts\Activate.ps1` instead.

Install dependencies and start the API:

```bash
python -m pip install -r requirements-dev.txt
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/docs> to test the API through Swagger UI. The `--reload` option is intended for local development.

## Run with Docker

Build the image from the repository root:

```bash
docker build -t ml-regression-api:0.1 .
```

Start the container:

```bash
docker run --rm -p 8000:8000 ml-regression-api:0.1
```

The API is then available at <http://localhost:8000/docs>.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Basic availability check |
| `POST` | `/predict` | Predict from a single numerical feature |
| `GET` | `/docs` | Interactive Swagger UI documentation |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5]}'
```

**Expected response:**

```json
{
  "status": "success",
  "input": [1.5],
  "prediction": 3.0
}
```

Invalid requests, such as `{"features": ["abc"]}` or `{"features": [1.5, 2.0]}`, return **HTTP 422** because they do not match the input schema.

## Automated tests and CI

Run the tests locally:

```bash
python -m pytest -v
```

The tests verify:

- `GET /health` returns a successful response.
- A valid input produces the expected prediction.
- A nonnumeric input is rejected.
- An incorrect number of features is rejected.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs pytest on pushes to `main` and pull requests targeting `main`.

**CI is separate from deployment:** Render can deploy from the connected GitHub branch independently of the GitHub Actions test results. The current exercise does not enforce a CI-gated deployment.

## Cloud deployment

The Dockerized API is hosted on **Render**:

- [API home](https://mldeploy-7gb3.onrender.com/)
- [Swagger UI](https://mldeploy-7gb3.onrender.com/docs)
- [Health check](https://mldeploy-7gb3.onrender.com/health)

The hosted service may take longer to respond after a period of inactivity, depending on the Render instance configuration.

## Key takeaways

Through this exercise, I practiced:

- Separating model inference logic from HTTP routes.
- Loading a saved model without retraining it.
- Validating API inputs with Pydantic.
- Packaging an ML application in a Docker image.
- Deploying an API to a public cloud endpoint.
- Detecting regressions through automated tests and CI.

## Scope and limitations

This repository is intentionally small and educational. It is a **learning exercise**, not a production-ready MLOps system. It does not include a training pipeline, model registry, authentication, advanced monitoring, retraining automation, or a deployment pipeline gated by CI. The model operates on a single numerical feature and synthetic training data and should not be used for real-world decisions.

Only load `joblib`/pickle model files from trusted sources: deserialization can execute code. The model should also be loaded with compatible dependency versions.

## Model attribution

The pretrained model used in this exercise comes from [Nuwantha2/ml-api-demo](https://github.com/Nuwantha2/ml-api-demo). I did **not** train this model as part of the exercise. My work focuses on the inference API, code organization, Docker packaging, cloud deployment, and automated tests. Reuse of upstream files is subject to the original repository's applicable license and terms.
