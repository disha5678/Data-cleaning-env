import pandas as pd

def remove_nulls(df, column):
    return df.dropna(subset=[column])

def fill_nulls(df, column, strategy="mean"):

    if strategy == "mean":
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())

    elif strategy == "median":
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].median())

    elif strategy == "mode":
        df[column] = df[column].fillna(df[column].mode()[0])

    return df

def convert_types(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def deduplicate(df):
    return df.drop_duplicates()

def trim_whitespace(df, column):
    df[column] = df[column].astype(str).str.strip()
    return df

def normalize_column(df, column):
    col = pd.to_numeric(df[column], errors='coerce')

    min_val = col.min()
    max_val = col.max()

    if max_val - min_val == 0:
        return df

    df[column] = (col - min_val) / (max_val - min_val)
    df[column] = df[column].fillna(0)  # optional safety

    return df
def compute_correlation(df):
    return df.corr(numeric_only=True)

def drop_correlated_feature(df, col):
    return df.drop(columns=[col])