from env.environment import DataCleaningEnv

env = DataCleaningEnv(task=2)
obs = env.reset()

print("Initial shape:", obs["shape"])

obs, r, _, _ = env.step({
    "type": "fill_nulls",
    "column": list(obs["dataset"].keys())[0],  # just testing
    "strategy": "mean"
})
print("After fill_nulls:", obs["shape"], "Reward:", r)
# Apply cleaning actions
obs, r, _, _ = env.step({"type": "remove_nulls", "column": "city"})
print("After remove_nulls:", obs["shape"], "Reward:", r)



obs, r, _, _ = env.step({
    "type": "normalize",
    "column": "income"
})

print("After normalization:", obs["shape"], "Reward:", r)


final = env.submit_cleaned_data(env.dirty_df)

print("Quality Score:", final["quality_score"])
print("Steps:", final["steps"])
print("Final Score:", final["final_score"])