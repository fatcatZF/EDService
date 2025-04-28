import numpy as np 

from environment_util import make_env

from stable_baselines3 import DQN

import argparse 

from predict_drift_score import *




parser = argparse.ArgumentParser()

parser.add_argument("--model-name", type=str, default="lof", 
                    help="name of the environment drift detection model")

args = parser.parse_args()




ONLINE_URL = "http://localhost:8000/online_predict_drift_score/cartpole"

#BATCH_URL = "http://localhost:8000/batch_predict_drift_score/cartpole"


# Create Environments
# env0: training environment
# env1: undrifted production environment
# env2: semantic-drifted environment
# env3: noisy observation environment
env0, env1, env2, env3 = make_env("cartpole")
print("Successfully create environments")



# load DQN agent
agent = DQN.load("./agents/dqn-cartpole.zip")
print("Successfully load trained agent.")



scores_12, scores_normalized_12 = online_predict_drift_score(env1, env2, 
                                                       agent, 
                                                       ONLINE_URL, 
                                                        model_name = "lof")



#scores_13, scores_normalized_13 = online_predict_drift_score(env1, env3, 
#                                                       agent, 
#                                                       ONLINE_URL, 
#                                                        model_name = "lof")



#scores_12, scores_normalized_12 = batch_predict_drift_score(env1, env2, 
#                                                       agent, 
#                                                       BATCH_URL, 
#                                                        model_name = "lof")



#scores_13, scores_normalized_13 = batch_predict_drift_score(env1, env3, 
#                                                      agent, 
#                                                       BATCH_URL, 
#                                                        model_name = "lof")














     
        