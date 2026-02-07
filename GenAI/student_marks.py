import numpy as np

students = np.array([
    [80, 70, 90],  # Student 1
    [60, 85, 75],  # Student 2
    [90, 88, 95]   # Student 3
])

weights = np.array([0.5, 0.3, 0.2])

final_scores = students @ weights
print('Final Scores: \n', final_scores)
