"""

Authors: Kiana Tarter & Francisco Herrera 
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Initialize generator 
gen = np.random.default_rng(seed=42)

# Sort data points into: large circle, medium circles, small dots
def sort_class(x,y):
    left_d = np.sqrt((x - 0.25)**2 + (y - 0.5)**2)
    right_d = np.sqrt((x - 0.75)**2 + (y - 0.5)**2)

    if (left_d < 0.1 or right_d < 0.1):
        return 2
    elif (left_d < 0.25 or (right_d > 0.25 and y > 0.5)):
        return 0
    else:
        return 1
        

size = 3000
per_class = size // 3
counts = [0,0,0]
points = []
labels =[]      

# Loop to generate points
while len(points) < size:
    x = gen.uniform(0,1)
    y = gen.uniform(0,1)

    from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)

    if from_center > 0.5:
        continue

    c = sort_class(x,y)

    if (counts[c] == per_class):
        continue

    points.append([x,y])
    labels.append(c)
    counts[c] = counts[c] + 1

print(len(points), counts)

# Plot the points
points = np.array(points)
labels = np.array(labels)

plt.scatter(points[:,0],points[:,1], c=labels)
plt.gca().set_aspect("equal")
plt.title("Yin Yang Dataset (3,000 samples)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("yinyang_samples.png", dpi=200)
plt.show()






