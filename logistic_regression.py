import sys
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Lasso
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
from numpy import array
import time
import pickle

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer

from image import Image
from log_loss import log_loss

def write(name, out):
	f = open(name, 'w')
	f.write(str(out))
	f.close()

def save(name, out):
	f = open(name, 'w')
	pickle.dump(out, f)
	f.close()

def load(name):
	f = open(name, 'r')
	new_net = pickle.load(f)
	return new_net

def test_unsupervised(model, test_data):
	preds = []
	for d in test_data:
		pred = model.activate(d)
		preds.append(pred)
	total_diff = 0.
	for i,p in enumerate(preds):
		target = test_data[i]
		difference = target - p
		sum_diff = float(sum(difference))
		root_squared = (sum_diff ** 2) ** 0.5
		avg_diff = root_squared/len(target)
		total_diff += avg_diff
	total_diff = total_diff/len(preds)
	return total_diff 
		


data = Image.data()
genders = Image.genders()

half = len(data)/2
train_data, train_genders = data[:half], genders[:half]
test_data, test_genders = data[half:], genders[half:]


net = buildNetwork(len(data[0]), len(data[0])/100, len(data[0]), outclass=SigmoidLayer)
ds = SupervisedDataSet(len(data[0]), len(data[0]))
for i,d in enumerate(train_data):
	if i < 250:
		t = train_genders[i]
		ds.addSample(d, d)

start = time.time()
trainer = BackpropTrainer(net, ds)
for i in range(100):
	trainer.train()


#model1 = LogisticRegression()
#model2 = ExtraTreesClassifier(n_estimators=30)
#model3 = GradientBoostingRegressor()
#model4 = KNeighborsRegressor(n_neighbors=20)
#model = GaussianNB()
#model = RandomForestClassifier(n_estimators=30)

all_preds = []
models = [net]
#models = [model2, model3, model4]
#confidences = [0.42, 0.50, 0.08]
confidences = [1.]
####
#for i, model in enumerate(models):
#	if hasattr(model, 'fit'):
#		model.fit(train_data, train_genders)
#	if hasattr(model, 'predict_proba'):
#		preds = [model.predict_proba(d)[0][1] for d in test_data]
#	elif hasattr(model, 'predict'):
#		preds = [model.predict(d)[0] for d in test_data]
#	else:
#		preds = [model.activate(d)[0] for d in test_data]
#
#	print preds[0:5]
#	preds = array(preds)
#	confidence = confidences[i]
#	weighted = preds * confidence
#	all_preds.append(weighted)
#
#
#merged = all_preds[0]
#for p in all_preds[1:]:
#	merged = merged + p
#####
#merged = merged/len(all_preds)

score = test_unsupervised(net, test_data)

#random = [0.5] * len(preds)

print "\n"
print "Unsupervised with 400 hidden nodes. 100 epochs."
print "Time: ", time.time() - start
print "Unsupervised score: ", score
#print "Score: ", log_loss(merged, test_genders)
#print "Random score: ", log_loss(random, test_genders)
#save('unsupervised_500_epoch.txt', net)
del(ds)
del(data)
write('unsupervised_400_hidden_100_epoch_score.txt', score)
#write('unsupervised_400_hidden_500_epoch_score.txt', log_loss(merged, test_genders))
