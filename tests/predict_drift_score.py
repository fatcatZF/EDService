import requests




def online_predict_drift_score(undrifted_env, drifted_env, agent, url, 
                   model_name = "lof"):
    
    # Query parameter sent to API endpoint
    params = {
        "model_name": model_name
    }

    undrifted_steps = 3000
    drifted_steps = 3000
    total_steps = undrifted_steps + drifted_steps

    scores = []
    scores_normalized = []

    env_current = undrifted_env
    obs_t, _ = env_current.reset()

    for t in range(1, total_steps+1):
        action_t, _state = agent.predict(obs_t, deterministic=True)

        obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)

        request_body = {
            "st": obs_t.tolist(),
            "at": [action_t.item()],
            "stplus1": obs_tplus1.tolist(),
            "rtplus1": [float(r_tplus1)],
            "t": t
        }

        response = requests.post(url, params=params, 
                                 json=request_body)


        
        response_data = response.json()

        score = response_data["drift_score"]
        score_normalized = response_data["drift_score_normalized"]

        scores.append(score)
        scores_normalized.append(score_normalized)


        done = terminated or truncated
        obs_t = obs_tplus1

        if done:
            obs_t, _ = env_current.reset()

        if t==undrifted_steps:
            env_current = drifted_env
            obs_t, _ = env_current.reset()

        
    return scores, scores_normalized