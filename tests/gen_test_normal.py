import numpy as np
from scipy.io import savemat


centers = np.random.uniform(-1,1, (10, 8))

clusters = []
for c in centers:
    clusters.append(np.random.normal(c, 0.3, 1000))

clusters = np.asarray(clusters)

savemat("clusters_norm_10_train.mat", {"X": clusters})

for i,c in enumerate(centers):
    savemat("clusters_norm_10_test_" + str(i) +".mat", {"X":np.random.normal(c, 0.3, 1000)})