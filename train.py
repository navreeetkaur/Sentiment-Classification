import sys
import numpy as np 
import pandas as pd
from preprocess import *
from features import *
from form_matrix import *
from predict import *
from variables import *

"""
1. ./compile.sh
2. ./train.sh trainfile.json devfile.json model_file
3. ./test.sh model_file testfile.json outputfile.txt
"""

##############################  VARIABLES ###########################################  
FILE_TRAIN = sys.argv[1]
FILE_DEV = sys.argv[2]
MODEL_FILE = sys.argv[3]

# clean, stem
# df = get_dataframe(FILE_TRAIN,train=True)
df_train = get_dataframe(FILE_TRAIN,train=True)
df_dev = get_dataframe(FILE_DEV,train=True)
df = pd.concat([df_train,df_dev], axis=0)

# count length of each review
df = count_length(df)

# features related to sentiment lexicons in reviews
df = get_sentiments(df,train=True,get_scores=get_scores)


# count different types of POS tags -----> fast without this
if TAG:
	df = run_parallel(df, count_POS_tags)

# get features from words
if use_tfidf:
	X_train_vocab,vectorizer_vocab = count_vocab_tfidf(df,params)
else:
	X_train_vocab,vectorizer_vocab = count_vocab(df,params)

# save vectorizer to pickle
pickle.dump(vectorizer_vocab, open(pickle_vocab_vector,'wb'))

if stack_features:
	X_train = get_feat_matrix(df,X_train_vocab)
else:
	X_train =X_train_vocab
	
Y_train = np.array(df['rating'])

# save model to pickle
if use_svr:
	model, svr = get_model2(X_train,Y_train)
	pickle.dump(model,open(MODEL_FILE,'wb'))
	pickle.dump(svr,open(MODEL_FILE+'_svr','wb'))
else:
	model = get_model(X_train,Y_train)
	pickle.dump(model, open(MODEL_FILE+'_svr','wb'))

