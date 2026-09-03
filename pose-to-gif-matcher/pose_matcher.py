import math


def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    A value closer to 1 means the vectors have similar directions.
    A value closer to 0 means they are less similar.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot_product = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        magnitude_a += a * a
        magnitude_b += b * b

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    # Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
