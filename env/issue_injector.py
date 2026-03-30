import numpy as np
import pandas as pd

def inject_issues(df):
    dirty = df.copy()
    manifest = {}

    # nulls
    idx = np.random.choice(len(df), 5, replace=False)
    dirty.loc[idx, "city"] = None
    manifest["nulls"] = idx.tolist()

    # duplicates
    dirty = pd.concat([dirty, dirty.iloc[:3]], ignore_index=True)
    manifest["duplicates"] = list(range(len(df), len(df)+3))

    # type issue
    dirty["age"] = dirty["age"].astype(str)

    return dirty, manifest