import numpy as np

def euclidean_distance(E1, E2):
    return np.sqrt(np.sum(np.square(np.array(E1) - np.array(E2))))

# Example VIT encodings (replace with your actual encodings)
vit_encoding1 = [0.2, 0.5, 0.8, 0.4, 0.1]
vit_encoding2 = [0.1, 0.6, 0.9, 0.2, 0.3]

distance = euclidean_distance(vit_encoding1, vit_encoding2)
print("Euclidean Distance:", distance)
