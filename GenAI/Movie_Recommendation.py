import numpy as np

movies = np.array([
    [5, 1, 4],  # Movie 0: Action-heavy Sci-Fi
    [1, 5, 1],  # Movie 1: Romantic
    [4, 1, 5],  # Movie 2: Action + Sci-Fi
    [0, 4, 0],  # Movie 3: Pure Romance
])

# User says:
#   Loves Action
#   Likes Sci-Fi
#   Hates Romance

user = np.array([5, 0, 4])
# This vector is:
#   “What kind of movie do I want?”

scores = movies @ user
# Movie 0 score
# = (5×5) + (1×0) + (4×4)
# = 25 + 0 + 16
# = 41

print(scores)
# [41  9 40  0]

recommended = np.argsort(scores)[::-1]
print(recommended)
 # [0 2 1 3]
Recommend in this order:
# Movie 0
# Movie 2
# Movie 1
# Movie 3

def cosine_similarity(A, b):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return A_norm @ b_norm

scores = cosine_similarity(movies, user)
print(scores)
# [0.98802352 0.27050089 0.96392539 0. ]


    # This is the same math used in:
    # Netflix
    # Spotify
    # Amazon
    # LLM embedding search
    # ChatGPT memory retrieval
