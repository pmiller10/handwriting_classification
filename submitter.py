from image import Image
from preprocess import Preprocess
from classifier import Classifier
from postprocess import PostProcess

genders = Image.genders()
all_data, ids = Image.all()
matrix = Preprocess.to_matrix(all_data)
matrix = Preprocess.remove_constants(matrix)
matrix = Preprocess.scale(matrix)
matrix = matrix.tolist()
train = matrix[:1128]
test = matrix[1128:]
test_ids = ids[1128:]
print len(train)
print len(test)
print len(test_ids)
print len(ids)
print len(matrix)
preds = Classifier.ensemble_preds(train, genders, test) # real
#preds = Classifier.ensemble_preds(train, genders, train) # fake

# for creating submission file
PostProcess.submission(test_ids, preds)
