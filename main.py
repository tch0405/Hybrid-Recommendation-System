from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from recommender import recommend_cf, recommend_cb, recommend_hybrid

app = FastAPI(title="Hybrid Recommender System")

class RecommendRequest(BaseModel):
    user_id: Optional[int] = None
    item_name: Optional[str] = None
    method: str  # 'cf', 'cb', or 'hybrid'
    top_n: int = 2

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.method == 'cf' and req.user_id is not None:
        return {"recommendations": recommend_cf(req.user_id, req.top_n)}
    elif req.method == 'cb' and req.item_name is not None:
        return {"recommendations": recommend_cb(req.item_name, req.top_n)}
    elif req.method == 'hybrid' and req.user_id is not None:
        return {"recommendations": recommend_hybrid(req.user_id, top_n=req.top_n)}
    else:
        return {"error": "Invalid input for the selected method."}
