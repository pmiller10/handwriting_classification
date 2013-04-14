from image import Image
from preprocess import Preprocess
from classifier import Classifier
from log_loss import log_loss

genders = Image.genders()
d = Image.data()
matrix = Preprocess.to_matrix(d)
print matrix.shape
matrix = Preprocess.remove_constants(matrix)
print matrix.shape
#matrix = Preprocess.scale(matrix)
#matrix = Preprocess.polynomial(matrix, 5)
matrix = Preprocess.scale(matrix)
print matrix.shape
matrix = matrix.tolist()
half = len(matrix)/2
train, cv = matrix[:half], matrix[half:]
train_genders, cv_genders = genders[:half], genders[half:]
preds = Classifier.ensemble_preds(train, train_genders, cv)
print "Score: ", log_loss(preds, cv_genders)
