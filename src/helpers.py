

from scipy.io import loadmat, savemat


def fetch_matlab_data(path):
    """
    Load a matlab file and return data training [and testing data]
    :param path: string
    :return: tuple
    """
    mat = loadmat(path)

    if "X" not in mat.keys():
        raise KeyError("There should be a X matrix in the file")

    return mat["X"]


