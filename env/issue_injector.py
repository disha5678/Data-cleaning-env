import numpy as np
import pandas as pd

def inject_issues(df):
    dirty = df.copy()
    manifest = {}
    rng = np.random.default_rng(42)


    # nulls
    idx = np.random.choice(len(df), 5, replace=False)
    dirty.loc[idx, "city"] = None
    columns = list(dirty.columns)
    selected_cols = rng.choice(columns, size=2, replace=False)
    null_indices = rng.choice(len(dirty), 20, replace=False).tolist()
    split = len(null_indices) // len(selected_cols)
    manifest["nulls"] = {}
    for i, col in enumerate(selected_cols):
        idxs = null_indices[i * split : (i + 1) * split]
        dirty.loc[idxs, col] = None
        manifest["nulls"][col] = idxs

    # duplicates
    dirty = pd.concat([dirty, dirty.iloc[:3]], ignore_index=True)
    manifest["duplicates"] = list(range(len(df), len(df)+3))

    # type issue
    dirty["age"] = dirty["age"].astype(str)

    return dirty, manifest