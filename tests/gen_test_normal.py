import numpy as np
from scipy.io import savemat
from sys import argv



if len(argv) < 5:
    print("Build a test set")
    print("python gen_test_normal.py num_clusters vector_dim standard_deviation clusters_size")
    exit(0)

centers = np.random.uniform(-1,1, (int(argv[1]), int(argv[2])))

clusters = []
for c in centers:
    clusters.append(np.random.normal(c, float(argv[3]), (int(argv[4]), int(argv[2]))))

clusters = np.asarray(clusters)

savemat("clusters_norm_10_train.mat", {"X": np.concatenate(clusters, axis=0)})

for i,c in enumerate(centers):
    savemat("clusters_norm_10_test_" + str(i) +".mat", {"X":np.random.normal(c, 0.3, (1000, 8))})