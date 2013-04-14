import sys
sys.path.append("/home/ubuntu/scikit-learn")
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import MultinomialNB
from numpy import array
import time

class Classifier():

    @classmethod
    def linear_preds(self, train, targets, cv):
        model = LinearRegression()
        model.fit(train, targets)
        preds = [model.predict(d)[0] for d in cv]
        return preds

    @classmethod
    def ensemble_preds(self, train, targets, cv):
        models = []
        #models.append(LinearRegression())
        models.append(LogisticRegression(C=10, penalty='l2', tol=1.0))
        models.append(svm.SVR(tol=0.001))
        #models.append(ExtraTreesClassifier(n_estimators=10))
        models.append(GradientBoostingRegressor())
        #models.append(KNeighborsRegressor(n_neighbors=20))
        #models.append(GaussianNB())
        #models.append(RandomForestClassifier(n_estimators=30))
        #models.append(Lasso())
        
        all_preds = []
        #confidences = [0.42, 0.50, 0.08]
        confidences = [1.0, 1.0, 1.0]
        ####
        for i, model in enumerate(models):
            if hasattr(model, 'fit'):
                model.fit(train, targets)
            if hasattr(model, 'predict_proba') and type(model) != svm.SVR:
                #preds = [model.predict(d)[0] for d in cv]
                preds = [model.predict_proba(d)[0][1] for d in cv]
            elif hasattr(model, 'predict'):
                preds = [model.predict(d)[0] for d in cv]
        #    else:
        #        preds = [model.activate(d)[0] for d in test_data]
        #
            print preds[0:5]
            preds = array(preds)
            confidence = confidences[i]
            weighted = preds * confidence
            all_preds.append(weighted)
        
    
        merged = all_preds[0]
        for p in all_preds[1:]:
            merged = merged + p
        merged = merged/len(all_preds)
        print merged[0:5]
        return self.vote(merged)

    @classmethod
    def vote(self, preds):
        ids_count = len(preds)/4 # count of unique ids
        ids = range(ids_count) # list with one element per author
        ids = array(ids)
        author_indexes = ids * 4
        scores = []
        print preds[0:12]
        for i in author_indexes:
            a,b,c,d = preds[i], preds[i+1], preds[i+2], preds[i+3] 
            scores.append((a + b + c + d)/4.)
        print scores[0:12]
        return scores
