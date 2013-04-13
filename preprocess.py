import numpy as np

class Preprocess():

    """ Accepts a list of lists and outputs a matrix """
    @classmethod
    def to_matrix(self, data):
        arrays = [np.array(d) for d in data[0:300]]
        return np.matrix(arrays)

    @classmethod
    def standard_deviation(self, matrix):
        return np.std(matrix, axis=0)

    """ accepts a numpy matrix and returns another numpy matrix with 
    all of the columns removed that have standard deviations of zero """
    @classmethod
    def remove_constants(self, matrix):
        std = np.std(matrix, axis=0)
        std = std.tolist()[0]
        nonzero_std_indices = [i for i,d in enumerate(std) if d != 0]
        return matrix[:,(nonzero_std_indices)]

#print Preprocess.remove_zero_indices()
