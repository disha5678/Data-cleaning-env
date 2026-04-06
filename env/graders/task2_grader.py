import numpy as np

def grade_task2(agent_df, clean_df):

    score = 0

    numeric_cols = agent_df.select_dtypes(include=np.number).columns

    # normalization check
    norm_score = 0
    for col in numeric_cols:
        if agent_df[col].min() >= 0 and agent_df[col].max() <= 1:
            norm_score += 1

    if len(numeric_cols) > 0:
        score += 0.4 * (norm_score / len(numeric_cols))

    # correlation reduction
    corr = agent_df.corr(numeric_only=True).abs()
    if (corr > 0.8).sum().sum() < len(corr):
        score += 0.3

    # data preserved
    if len(agent_df) <= len(clean_df):
        score += 0.3

    return round(min(score, 1.0), 4)