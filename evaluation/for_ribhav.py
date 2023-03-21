
import pickle as pkl
import numpy as np

# save input into a file

fixation_points = np.random.randn((100, 2))
pkl.dump(fixation_points, open("./points.pkl", "wb"))