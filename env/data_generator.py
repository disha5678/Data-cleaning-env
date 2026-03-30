import pandas as pd
import numpy as np
from faker import Faker

def generate_clean_dataset(n_rows=50, seed=42):
    fake = Faker("en_IN")
    np.random.seed(seed)

    df = pd.DataFrame({
        "customer_id": [f"C-{1000+i}" for i in range(n_rows)],
        "name": [fake.name() for _ in range(n_rows)],
        "age": np.random.randint(18, 60, n_rows),
        "city": [fake.city() for _ in range(n_rows)],
        "income": np.random.randint(20000, 100000, n_rows)
    })

    return df