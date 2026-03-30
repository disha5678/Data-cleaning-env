from env.environment import DataCleaningEnv

env = DataCleaningEnv()
obs = env.reset()

print("Initial dataset:")
print(obs["dataset"].head())

obs, reward, done, _ = env.step({
    "type": "inspect_column",
    "column": "age"
})

print("Reward:", reward)