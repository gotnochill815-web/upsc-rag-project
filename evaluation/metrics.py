def recall_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    hits = len(set(retrieved) & set(relevant))

    if len(relevant) == 0:
        return 0

    return hits / len(relevant)


def precision_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    hits = len(set(retrieved) & set(relevant))

    return hits / k


def reciprocal_rank(retrieved, relevant):

    for i, doc in enumerate(retrieved):

        if doc in relevant:
            return 1 / (i + 1)

    return 0