from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os 



app = FastAPI(title="Environment Drift Detection Service")



class SATransition(BaseModel):
    st: list[float] # current state s_t
    at: list[float] | int # action a_t 
    stplus1: list[float] # next state s_t


