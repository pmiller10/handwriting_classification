from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from numpy import array
import time

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer

from image import Image
from log_loss import log_loss

start = time.time()

data = Image.data()
genders = Image.genders()

half = len(data)/2
train_data, train_genders = data[:half], genders[:half]
test_data, test_genders = data[half:], genders[half:]


net = buildNetwork(len(data[0]), 50, 5, 1, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)
ds = SupervisedDataSet(len(data[0]), 1)
for i,d in enumerate(train_data):
	t = train_genders[i]
	ds.addSample(d, t)

trainer = BackpropTrainer(net, ds)
for i in range(10):
	trainer.train()


model1 = LogisticRegression()
model2 = ExtraTreesClassifier(n_estimators=30)
model3 = GradientBoostingRegressor()
model4 = KNeighborsRegressor(n_neighbors=20)
#model = GaussianNB()
#model = RandomForestClassifier(n_estimators=30)

all_preds = []
models = [net]
#models = [model2, model3, model4]
#confidences = [0.42, 0.50, 0.08]
confidences = [1.]
for i, model in enumerate(models):
	if hasattr(model, 'fit'):
		model.fit(train_data, train_genders)
	if hasattr(model, 'predict_proba'):
		preds = [model.predict_proba(d)[0][1] for d in test_data]
	elif hasattr(model, 'predict'):
		preds = [model.predict(d)[0] for d in test_data]
	else:
		preds = [model.activate(d)[0] for d in test_data]

	print preds[0:5]
	preds = array(preds)
	confidence = confidences[i]
	weighted = preds * confidence
	all_preds.append(weighted)


merged = all_preds[0]
for p in all_preds[1:]:
	merged = merged + p

#merged = merged/len(all_preds)

random = [0.5] * len(preds)

print log_loss(merged, test_genders)
print log_loss(random, test_genders)

print time.time() - start
