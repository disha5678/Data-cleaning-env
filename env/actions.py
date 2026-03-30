import pandas as pd

def remove_nulls(df, column):
    return df.dropna(subset=[column])


def convert_types(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def deduplicate(df):
    return df.drop_duplicates()