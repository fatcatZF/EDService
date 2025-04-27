import numpy as np 

from environment_util import make_env

from stable_baselines3 import PPO 


import argparse


from predict_drift_score import *


parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str, default="lof", 
                    help="name of the environment drift detection model")

args = parser.parse_args()


ONLINE_URL = "http://localhost:8000/online_predict_drift_score/lunarlander"

BATCH_URL = "http://localhost:8000/batch_predict_drift_score/lunarlander"



env0, env1, env2, env3 = make_env("lunarlander")
print("Successfully create environments")


agent = PPO.load("./agents/ppo-lunarlander.zip")
print("Successfully load trained agent.")



scores_12, scores_normalized_12 = online_predict_drift_score(env1, env2, 
                                                       agent, 
                                                       ONLINE_URL, 
                                                        model_name = args.model_name)



scores_13, scores_normalized_13 = online_predict_drift_score(env1, env3, 
                                                       agent, 
                                                       ONLINE_URL, 
                                                        model_name = args.model_name)



scores_12, scores_normalized_12 = batch_predict_drift_score(env1, env2, 
                                                       agent, 
                                                       BATCH_URL, 
                                                        model_name = args.model_name)



scores_13, scores_normalized_13 = batch_predict_drift_score(env1, env3, 
                                                       agent, 
                                                       BATCH_URL, 
                                                        model_name = args.model_name)



