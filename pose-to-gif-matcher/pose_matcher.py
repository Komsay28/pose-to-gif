import math
import json


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

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def load_database(json_path):
    """
    Load the pre-processed GIF pose database from poses.json.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def best_match_for_gif(live_pose, gif_entry):
    """
    Given a live pose vector and one GIF's database entry
    (which contains a list of pose vectors, one per sampled frame),
    return the highest similarity score found across all its frames.
    """
    best_score = -1.0

    for frame_pose in gif_entry["poses"]:
        score = cosine_similarity(live_pose, frame_pose)
        if score > best_score:
            best_score = score

    return best_score


def find_best_match(live_pose, database, top_n=1):
    """
    Compare a live pose against every GIF in the database.

    Returns a list of (gif_entry, score) tuples, sorted by
    score descending, limited to top_n results.
    """
    results = []

    for gif_entry in database:
        score = best_match_for_gif(live_pose, gif_entry)
        results.append((gif_entry, score))

    results.sort(key=lambda pair: pair[1], reverse=True)

    return results[:top_n]