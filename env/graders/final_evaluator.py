def compute_final_score(quality_score, steps):

    efficiency_score = max(0, 1 - (steps / 20))

    final_score = 0.75 * quality_score + 0.25 * efficiency_score

    return round(final_score, 4)