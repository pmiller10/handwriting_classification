from image import Image
from preprocess import Preprocess
import numpy as np

d = Image.data()
matrix = Preprocess.to_matrix(d)
print matrix.shape
matrix = Preprocess.remove_constants(matrix)
print len(matrix.tolist()[0])
