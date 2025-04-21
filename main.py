from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os 

ed_models = {}

app = FastAPI(title="Environment Drift Detection Service")



class SATransition(BaseModel):
    st: list[float] # current state s_t
    at: list[float] | int # action a_t 
    stplus1: list[float] # next state s_t


@app.get("/")
async def root():
    return {
        "service": "Environment Drift Detection API",
        "status": "running",
        "endpoints": ["/", "/get_drift_score/"]
    }


@app.post("/get_drift_score/{env_name}")
async def get_drift_score(*,
                          env_name:str="cartpole", 
                          ed_model:str = "lof",
                          transition:SATransition):
    if ed_models.get(f"{env_name}") is not None:
        if ed_models.get(f"{env_name}").get(f"{ed_model}") is not None:
            ed_model = ed_models.get(f"{env_name}").get(f"{ed_model}")
            return {"status": "Model has been loaded."}

    else:
        return {"status": "Environment is not supported."}
    