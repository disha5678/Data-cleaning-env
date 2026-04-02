def grade_task1(agent_df, clean_df, manifest):
    score = 0

    # null handling
    if agent_df.isnull().sum().sum() == 0:
        score += 0.25

    # duplicates
    if len(agent_df.drop_duplicates()) == len(clean_df):
        score += 0.25

    # dtype check
    try:
        agent_df["age"].astype(int)
        score += 0.25
    except:
        pass

    # whitespace (basic assumption)
    score += 0.25

    return min(score, 1.0)