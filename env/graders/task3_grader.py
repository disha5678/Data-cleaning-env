def grade_task3(agent_df, clean_df):

    score = 0

    # ✅ 1. Data preservation (VERY IMPORTANT)
    ratio = len(agent_df) / len(clean_df)

    if ratio > 0.9:
        score += 0.3
    elif ratio > 0.75:
        score += 0.2
    elif ratio > 0.6:
        score += 0.1

    # ✅ 2. Null removal
    nulls = agent_df.isnull().sum().sum()
    if nulls == 0:
        score += 0.25

    # ✅ 3. Duplicate removal
    if len(agent_df) == len(agent_df.drop_duplicates()):
        score += 0.2

    # ✅ 4. Structure preservation (columns)
    col_diff = abs(agent_df.shape[1] - clean_df.shape[1])
    if col_diff == 0:
        score += 0.15
    elif col_diff <= 1:
        score += 0.1

    # ✅ 5. No excessive cleaning (penalty)
    if ratio < 0.5:
        score -= 0.2   # too much data loss

    return round(max(min(score, 1.0), 0), 4)