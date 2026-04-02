import pandas as pd
from env.actions import remove_nulls

# Step 1: Create sample dataset
df = pd.DataFrame({
    "city": ["Delhi", None, "Mumbai", None]
})

print("Original Data:")
print(df)

# Step 2: Apply your function
cleaned_df = remove_nulls(df, "city")

# Step 3: Print result
print("\nAfter remove_nulls:")
print(cleaned_df)