from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel

import shutil

import joblib
import os 
import json




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
    at: list[float] | int # action a_t 
    stplus1: list[float] # next state s_t







@app.get("/")
async def root():
    return {
        "service": "Environment Drift Detection API",
        "status": "running",
        "endpoints": ["/", "/upload-edmodel", "/get_drift_score/"]
    }





@app.post("/upload-edmodel")
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
    

        
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
