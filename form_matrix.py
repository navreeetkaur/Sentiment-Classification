import pickle
import numpy as np 
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from variables import *
from utils import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def count_vocab(data,params):
	reviews = data['review'].apply(lambda x: " ".join(x))
	if form_vocab:
		counter = get_counter(data['review'])
		vocab = build_vocab_size(counter, vocab_size) # can change vocabulary generation method
		vectorizer_vocab = CountVectorizer(vocabulary=vocab, binary=params['binary']) # binary, max_df, min_df
	else:
		vectorizer_vocab = CountVectorizer(**params) # binary, max_df, min_df
	X_vocab = (vectorizer_vocab.fit_transform(reviews)).toarray()
	return X_vocab, vectorizer_vocab	

def count_vocab_test(data,vectorizer):
	reviews = data['review'].apply(lambda x: " ".join(x))
	X_vocab = (vectorizer.transform(reviews)).toarray()
	return X_vocab

def count_vocab_tfidf(data,params):
	reviews = data['review'].apply(lambda x: " ".join(x))
	if form_vocab:
		counter = get_counter(data['review'])
		vocab = build_vocab_size(counter, vocab_size) # can change vocabulary generation method
		vectorizer_vocab = TfidfVectorizer(vocabulary=vocab) # binary, max_df, min_df
	else:
		vectorizer_vocab = TfidfVectorizer(**params) # binary, max_df, min_df
	X_vocab = (vectorizer_vocab.fit_transform(reviews)).toarray()
	return X_vocab, vectorizer_vocab


def get_feat_matrix(data, X_vocab):
	num_caps = np.array(data['capitals'].apply(lambda x: len(x)))
	num_caps= num_caps.reshape(len(num_caps),1)
	X = num_caps
	dim0 = data.shape[0]
	for column in data:
	    if column in not_features:
	        continue
	    X = np.concatenate((X, np.array(data[column]).reshape(dim0,1)), axis=1)
	X = np.concatenate((X_vocab,X),axis=1)
	print(f'Shape of X: {X.shape}')
	return X