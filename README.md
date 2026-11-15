# Heart Disease Prediction API

FastAPI service that serves a scikit-learn classifier (StandardScaler + Random Forest)
trained on the Kaggle Heart Disease dataset. Dockerized and deployed on Render.

**Live URL:** https://YOUR-SERVICE.onrender.com  (Swagger UI at `/docs`)

## Project structure
```
app/            FastAPI app (main.py, schemas.py)
model/          heart_model.joblib (trained model)
data/           heart.csv (Kaggle dataset, not used at runtime)
train.py        training script
Dockerfile
docker-compose.yml
requirements.txt
```

## Train the model
1. Download `heart.csv` from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset into `data/`.
2. `pip install -r requirements.txt && python train.py`

## Run with Docker
```
docker-compose build
docker-compose up
```
Open http://localhost:8000/docs

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | /health | Service status |
| GET | /info | Model type and feature list |
| POST | /predict | Returns `heart_disease: true/false` and probability |

### Example
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":0,"trestbps":140,"chol":240,"fbs":0,"restecg":1,"thalach":120,"exang":1,"oldpeak":2.0,"slope":1,"ca":2,"thal":3}'
```

## Deploy on Render
1. Push this repo to GitHub (include `model/heart_model.joblib`).
2. Render dashboard -> New -> Web Service -> connect the repo.
3. Language/Environment: **Docker**, root directory = repo root.
4. Deploy, then test `/health` and `/predict` with the live URL.

*Educational project; not for medical use.*
