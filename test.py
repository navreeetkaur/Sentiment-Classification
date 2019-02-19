import sys
import tqdm
import numpy as np 
import pandas as pd
import pickle
from preprocess import *
from features import *
from form_matrix import *
from variables import *
from predict import *

"""
1. ./compile.sh
2. ./train.sh trainfile.json testfile.json model_file
3. ./test.sh model_file testfile.json outputfile.txt
"""

##############################  VARIABLES ###########################################  
MODEL_FILE = sys.argv[1]
FILE_TEST = sys.argv[2]
FILE_OUT = sys.argv[3]

with open(pickle_vocab_vector,'rb') as f:
	vectorizer_vocab = pickle.load(f)

# clean, stem
df_test = get_dataframe(FILE_TEST,train=False)

# count length of each review
df_test = count_length(df_test)

# features related to sentiment lexicons in reviews
df_test = get_sentiments(df_test,train=False,get_scores=get_scores)

# count different types of POS tags -----> fast without POS tagging
if TAG:
	df_test = run_parallel(test, count_POS_tags)

X_test_vocab = count_vocab_test(df_test,vectorizer_vocab)
	
if stack_features:
	X_test = get_feat_matrix(df_test,X_test_vocab)
else:
	X_test = X_test_vocab

if use_svr:
	model = pickle.load(open(MODEL_FILE,'rb')) 
	svr = pickle.load(open(MODEL_FILE+'_svr','rb')) 
	y_pred = get_prediction_model2(model=model, svr=svr, X_test=X_test)
else:
	model = pickle.load(open(MODEL_FILE,'rb')) 
	y_pred = get_prediction_model(model,X_test)

with open(FILE_OUT,'w') as f:
	for prediction in y_pred:
		if prediction<1.0:
			prediction=1.0
		if prediction>5.0:
			prediction=5.0
		f.write(str(prediction)+'\n')

