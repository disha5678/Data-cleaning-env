def grade_task3(agent_df, clean_df):

    score = 0

    # data preservation
    ratio = len(agent_df) / len(clean_df)

    if ratio > 0.8:
        score += 0.3
    elif ratio > 0.6:
        score += 0.2

    # no nulls
    if agent_df.isnull().sum().sum() == 0:
        score += 0.3

    # no duplicates
    if len(agent_df) == len(agent_df.drop_duplicates()):
        score += 0.2

    # structure maintained
    if abs(agent_df.shape[1] - clean_df.shape[1]) <= 1:
        score += 0.2

    return round(min(score, 1.0), 4)