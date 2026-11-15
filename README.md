# Heart Disease Prediction API (FastAPI + Docker)

FastAPI service serving a Random Forest classifier trained on the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

## Layout
- `app/` – FastAPI app (`main.py`) and Pydantic schemas (`schemas.py`)
- `model/heart_model.joblib` – trained model (created by `train.py` from `data/heart.csv`)
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`

## Run locally
```
docker-compose build
docker-compose up
```
Swagger UI: http://localhost:8000/docs

## Endpoints
- `GET /health`
- `GET /info` – model type and feature list
- `POST /predict` – returns `{"heart_disease": true|false}`

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

## Retrain
`pip install -r requirements.txt && python train.py`

## Deploy on Render
New Web Service → connect this GitHub repo → Language: Docker (root Dockerfile).
The container listens on Render's `$PORT` (defaults to 8000).

Live URL: _add after deploying_
