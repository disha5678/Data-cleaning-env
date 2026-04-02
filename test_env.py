import pandas as pd
from env.actions import remove_nulls

# Step 1: Create sample dataset
df = pd.DataFrame({
    "city": ["Delhi", None, "Mumbai", None]
})

print("Initial dataset:")
print(obs["dataset"].head())

obs, reward, done, _ = env.step({
    "type": "inspect_column",
    "column": "age"
})

print("Reward:", reward)