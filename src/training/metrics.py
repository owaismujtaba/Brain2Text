"""Evaluation metrics."""


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance in words, divided by the reference length.

    A value of 0 means a perfect match; 1 means every word is wrong.
    """
    ref, hyp = reference.split(), hypothesis.split()

    # dist[i][j] = edit distance between ref[:i] and hyp[:j]
    dist = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dist[i][0] = i
    for j in range(len(hyp) + 1):
        dist[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,       # deletion
                dist[i][j - 1] + 1,       # insertion
                dist[i - 1][j - 1] + cost,  # match / substitution
            )

    return dist[len(ref)][len(hyp)] / max(len(ref), 1)
