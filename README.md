🔀 Hybrid Recommender System API
A FastAPI-based recommendation engine that supports:

## 🧑‍🤝‍🧑 Collaborative Filtering (user-user)

## 🔍 Content-Based Filtering (item metadata)

## ⚖️ Hybrid Filtering (weighted combination)

## 🚀 Getting Started
```bash

pip install -r requirements.txt
uvicorn main:app --reload

```
Visit Swagger UI: http://127.0.0.1:8000/docs

📡 API: POST /recommend
Example request:


Edit
{
  "user_id": 1,
  "item_name": "A",
  "method": "hybrid",
  "top_n": 5
}