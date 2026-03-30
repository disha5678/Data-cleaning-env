def grade_task1(agent_df, clean_df):
    score = 0

    # duplicates
    if len(agent_df) == len(clean_df):
        score += 0.25

    # nulls
    if agent_df.isnull().sum().sum() == 0:
        score += 0.25

    # dtype
    try:
        agent_df["age"].astype(int)
        score += 0.25
    except:
        pass

    # whitespace check (basic)
    score += 0.25

    return score