from image import Image
from preprocess import Preprocess
from classifier import Classifier
from log_loss import log_loss
from postprocess import PostProcess

genders = Image.genders()
d, _ = Image.data()
matrix = Preprocess.to_matrix(d)
print matrix.shape
matrix = Preprocess.remove_constants(matrix)
print matrix.shape
matrix = Preprocess.scale(matrix)
matrix = Preprocess.polynomial(matrix, 3)
matrix = Preprocess.scale(matrix)
print matrix.shape
matrix = matrix.tolist()
half = len(matrix)/2
train, cv = matrix[:half], matrix[half:]
train_genders, cv_genders = genders[:half], genders[half:]
cv_genders = cv_genders[0::4]
preds = Classifier.ensemble_preds(train, train_genders, cv)
print "Score: ", log_loss(preds, cv_genders)
