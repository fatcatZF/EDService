from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel

import shutil

import joblib
import os 
import json

import numpy as np 




ed_models = {}




app = FastAPI(title="Environment Drift Detection Service")


REGISTRY_PATH = "./model_registry.json"


@app.on_event("startup")
def load_model_registry():
    global model_registry 
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                model_registry = json.load(f)
        except json.JSONDecodeError:
            print("model_registry.json is invalid. Starting with empty registry.")
            model_registry = {}
            
    else:
        model_registry = {}
    print("Model registry loaded.")



@app.on_event("shutdown")
def save_model_registry():
    with open(REGISTRY_PATH, "w") as f:
        json.dump(model_registry, f)
    print("💾 Model registry saved.")





class SASingleTransition(BaseModel):
    # A single transition of a Single Agent RL Environment
    st: list[float] # current state s_t
    at: list[float]  # action a_t 
    stplus1: list[float] # next state s_t







@app.get("/")
async def root():
    return {
        "service": "Environment Drift Detection API",
        "status": "running",
        "endpoints": ["/", "/upload-edmodel", "/get_drift_score/"]
    }





@app.post("/upload-edmodel/")
async def upload_edmodel(
    env: str,
    model_folder: str,
    model: UploadFile,
    norm_config_file :UploadFile|None = None
):
    
    env_dir = os.path.join("./ed-models", env)

    # create folder if not exist
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)

    # save the uploaded file
    model_folder_path = os.path.join(env_dir, model_folder)
    os.makedirs(model_folder_path)

    model_path = os.path.join(model_folder_path, "model.joblib")
    with open(model_path, 'wb') as buffer:
        shutil.copyfileobj(model.file, buffer)

    if norm_config_file is not None:
        config_path = os.path.join(model_folder_path, "norm_config.json")
        with open(config_path, 'wb') as buffer:
            shutil.copyfileobj(norm_config_file.file, buffer)

    # register model
    if env not in model_registry:
        model_registry[env] ={}
    
    model_registry[env][model_folder] = {
        "model_path": model_path,
    }

    if norm_config_file is not None:
        model_registry[env][model_folder]["norm_config_path"] = config_path


    return {
        "message": f"Model {model_folder} uploaded to environment {env}"
    }
    




@app.post("/online_predict_drift_score/{env}")
async def online_predict_drift_score(
    env: str,
    model_name: str,
    transition: SASingleTransition
):
    response = {}
    # Check if the environment is supported and the model exists
    if model_registry.get(env, None) is None:
        return {"info": f"There are no ed models for {env}"}
    
    model_folder_info = model_registry[env].get(model_name, None)

    if model_folder_info is None:
        return {"info": f"No f{model_name} for f{env}"}
    
    model_path = model_folder_info["model_path"]
    norm_config_path = model_folder_info.get("norm_config_path", None)
    

    # Check if the model has been loaded in memory
    model = ed_models.get(f"{env}_{model_name}_model", None)
    mu = ed_models.get(f"{env}_{model_name}_mu", None)
    sigma = ed_models.get(f"{env}_{model_name}_sigma", None)

    if model is None:
        model = joblib.load(model_path)
        ed_models[f"{env}_{model_name}_model"] = model 
        # Check if the normalization configuration exisits
        #norm_config_path = os.path.join(model_folder_path, "norm_config.json")
        
        if norm_config_path is not None:
            with open(norm_config_path, 'r') as f:
                norm_config = json.load(f) 
            mu = norm_config["mu"]
            sigma = norm_config["sigma"]
            ed_models[f"{env}_{model_name}_mu"] = mu 
            ed_models[f"{env}_{model_name}_sigma"] = sigma

    st = np.array(transition.st)
    stplus1 = np.array(transition.stplus1)
    at = np.array(transition.at)
    x = np.concatenate([st, stplus1-st]).reshape(1, -1)
    x = np.concatenate([x, at.reshape(1, -1)], axis=1).astype(np.float32)

    drift_score = - model.decision_function(x)[0]
    
    response["drift_score"] = drift_score

    if mu is not None and sigma is not None:
        response["drift_score_normalized"] = (drift_score-mu)/(sigma+1e-6)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
