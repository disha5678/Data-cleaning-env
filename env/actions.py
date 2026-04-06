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