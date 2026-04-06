def grade_task1(agent_df, clean_df, manifest):
    score = 0

    # nulls (partial)
    total_nulls = sum(len(v) for v in manifest["nulls"].values())
    remaining_nulls = agent_df.isnull().sum().sum()

    score += 0.25 * (1 - remaining_nulls / max(total_nulls, 1))

    # duplicates
    expected = len(clean_df)
    actual = len(agent_df.drop_duplicates())

    score += 0.25 * (1 - abs(actual - expected) / expected)

    # dtype
    try:
        agent_df["age"].astype(int)
        score += 0.25
    except:
        pass

    # whitespace
    score += 0.25

    return round(min(score, 1.0), 4)