import numpy as np

documents = [
    "Interstellar is a science fiction movie about space and time.",
    "Inception is a sci-fi thriller that explores dreams within dreams.",
    "Titanic is a romantic drama set on a sinking ship.",
    "Avengers Endgame is a superhero action movie.",
    "The Hangover is a comedy about friends in Las Vegas.",
    "Guardians of the Galaxy is a sci-fi action comedy."
]

# [Action, Romance, Sci-Fi, Comedy, Drama]
doc_embeddings = np.array([
    [1, 0, 5, 0, 4],  # Interstellar
    [2, 0, 5, 0, 3],  # Inception
    [0, 5, 0, 0, 5],  # Titanic
    [5, 0, 1, 0, 2],  # Avengers
    [0, 0, 0, 5, 1],  # Hangover
    [4, 0, 4, 5, 2]   # Guardians
])

query_embedding = np.array([5, 0, 5, 0, 0])

def cosine_similarity(A, q):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    q_norm = q / np.linalg.norm(q)
    return A_norm @ q_norm

scores = cosine_similarity(doc_embeddings, query_embedding)
top_k = 2
top_indices = np.argsort(scores)[::-1][:top_k]

retrieved_docs = [documents[i] for i in top_indices]
for doc in retrieved_docs:
    print("-", doc)





context = "\n".join(retrieved_docs)

prompt = f"""
You are a movie recommendation assistant.

Context:
{context}

Question:
Recommend a good sci-fi action movie and explain why.
"""


# SIMULATED LLM
def generate_answer(prompt):
    return (
        "Based on the context, Guardians of the Galaxy is a great sci-fi action movie. "
        "It combines futuristic elements with intense action and humor, making it entertaining "
        "while still delivering a strong science fiction experience."
    )

answer = generate_answer(prompt)
print(answer)
