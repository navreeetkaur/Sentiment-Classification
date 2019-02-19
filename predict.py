import numpy as np

from sklearn.metrics import *

from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

from utils import *
from variables import *


def get_model(X_train,Y_train):
	model = LogisticRegression(penalty='l2',C=1.0,n_jobs=-1)
	model.fit(X_train, Y_train)
	return model

def get_model2(X_train,Y_train):
	model = LogisticRegression(penalty='l2',C=1.0,n_jobs=-1)
	model.fit(X_train, Y_train)
	y_pred = model.predict_proba(X_train)
	svr = SVR(kernel='linear')
	svr.fit(y_pred, Y_train)
	return model,svr
	
def get_prediction_model(model, X_test):
	y_score = model.predict_proba(X_test)
	categories=np.array([1.0,2.0,3.0,4.0,5.0])
	y_pred = np.matmul(y_score,categories)
	return y_pred

def get_prediction_model2(model, svr, X_test):
	print(f'\nMODEL:')
	print(model)
	print(f'\nSVR:')
	print(svr)
	print(f'Log reg feature size: {X_test.shape[1]}')
	y_score = model.predict_proba(X_test)
	print(f'SVR feature size: {y_score.shape[1]}')
	y_pred = svr.predict(y_score)
	return y_pred



	