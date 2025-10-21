from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import uvicorn

app = FastAPI()

class UserRequest(BaseModel):
    user_id: int

# Load model once at startup
model = mlflow.sklearn.load_model("mlflow/mlruns/model")

@app.post("/recommend")
def recommend_movies(req: UserRequest):
    try:
        recs = model.recommend(req.user_id)
        return {"user_id": req.user_id, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
