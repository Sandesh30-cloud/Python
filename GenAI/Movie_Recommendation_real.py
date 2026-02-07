import numpy as np

movies = [
    "Avengers: Endgame",
    "Titanic",
    "Interstellar",
    "The Hangover",
    "Inception",
    "The Notebook",
    "Guardians of the Galaxy"
]
movie_features = np.array([
    [5, 1, 4, 2],  # Avengers
    [1, 5, 1, 1],  # Titanic
    [3, 1, 5, 1],  # Interstellar
    [1, 1, 0, 5],  # Hangover
    [4, 1, 5, 1],  # Inception
    [0, 5, 0, 1],  # Notebook
    [4, 1, 4, 5]   # Guardians
])

user_preference = np.array([5, 0, 5, 3])
# [Action, Romance, Sci-Fi, Comedy]

scores = movie_features @ user_preference

for movie, score in zip(movies, scores):
    print(f"{movie}: {score}")


ranking = np.argsort(scores)[::-1]
print("\n🎯 RECOMMENDED MOVIES (BEST → WORST):\n")
for i in ranking:
    print(movies[i])
    
    
def cosine_similarity(A, b):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return A_norm @ b_norm

scores = cosine_similarity(movie_features, user_preference)
ranking = np.argsort(scores)[::-1]

print("\n🎯 COSINE-SIMILARITY RECOMMENDATIONS:\n")
for i in ranking:
    print(movies[i], "→", round(scores[i], 3))


