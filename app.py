from fastapi import FastAPI
from env.environment import DataCleaningEnv

app = FastAPI()   # 🔥 MUST BE BEFORE @app
env = DataCleaningEnv(task=1)

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    obs, reward, done, _ = env.step(action)
    return {"obs": obs, "reward": reward, "done": done}

@app.get("/state")
def state():
    return env.state()