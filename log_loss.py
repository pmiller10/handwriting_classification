import scipy as sp
from math import log


def log_loss(predicted, target):
	if len(predicted) != len(target):
		print 'lengths not equal!'
		return
	#print target
	target = [float(x) for x in target]   # make sure all float values
	#print target
	#print predicted 
	predicted = [min([max([x,1e-15]),1-1e-15]) for x in predicted]  # within (0,1) interval
	#print predicted 
	return -(1.0/len(target))*sum([target[i]*log(predicted[i]) + (1.0-target[i])*log(1.0-predicted[i]) for i in xrange(len(target))])
