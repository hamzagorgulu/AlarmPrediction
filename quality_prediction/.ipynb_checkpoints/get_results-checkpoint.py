
import numpy as np
f = open("grid_search_results.json", "rb")
data = np.load(f, allow_pickle=True)
print(data)
