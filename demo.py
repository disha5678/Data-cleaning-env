from env.environment import DataCleaningEnv

# 🔥 choose task here
env = DataCleaningEnv(task=2)

obs = env.reset()

print("Initial dataset shape:", obs["shape"])

# simulate agent actions
actions = [
    {"type": "fill_nulls", "column": "age"},
    {"type": "normalize", "column": "age"},
    {"type": "deduplicate"}
]

for action in actions:
    obs, reward, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

# final evaluation
final = env.submit_cleaned_data(env.dirty_df)

print("\nFinal Results:")
print("Quality Score:", final["quality_score"])
print("Steps:", final["steps"])
print("Final Score:", final["final_score"])