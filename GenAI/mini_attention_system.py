import numpy as np

movies = [
    "Avengers: Endgame",
    "Titanic",
    "Interstellar",
    "The Hangover",
    "Inception",
    "Guardians of the Galaxy"
]

movie_features = np.array([
    [5, 1, 4, 2],  # Avengers
    [1, 5, 1, 1],  # Titanic
    [3, 1, 5, 1],  # Interstellar
    [1, 1, 0, 5],  # Hangover
    [4, 1, 5, 1],  # Inception
    [4, 1, 4, 5]   # Guardians
])
#   Each row = movie
#   Each column = [Action, Romance, Sci-Fi, Comedy]

# User says
# “I’m in a Sci-Fi + Action mood today”
query = np.array([5, 0, 5, 0])

keys = movie_features.copy()
values = movie_features.copy()

scores = keys @ query
print('Scores:\n ',scores)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
attention_weights = softmax(scores)
print('\nattention_weights:\n ', attention_weights)

output = attention_weights @ values
print('Output:\n',output)
# It answers:
# “Given my mood, what kind of movie should I watch?”

ranking = np.argsort(scores)[::-1]

print("\n🎯 ATTENTION-BASED RECOMMENDATIONS:\n")
for i in ranking:
    print(movies[i])


            
            | Transformer Term | What I Did        |
            | ---------------- | ----------------- |
            | Query (Q)        | User mood         |
            | Key (K)          | Movie description |
            | Value (V)        | Movie info        |
            | Q·Kᵀ             | Relevance         |
            | Softmax          | Focus             |
            | Weighted sum     | Context output    |
