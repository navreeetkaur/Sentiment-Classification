import pickle
import numpy as np 
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from knowledge import STOPWORDS, get_lexicon_dictionary
from preprocess import run_parallel
from utils import *
from variables import *


stemmed_POSITIVES, stemmed_NEGATIVES, pos_scores, neg_scores = get_lexicon_dictionary(get_scores=get_scores)

if form_vocab:
    pickle.dump(stemmed_POSITIVES, open(stemmed_pos_file,'wb'))
    pickle.dump(stemmed_NEGATIVES, open(stemmed_neg_file,'wb'))

def count_num_lexicon(data_row, lexicon_list, name):
    review = data_row['review']
    capitals = data_row['capitals']
    count = 0
    not_counts = 0
    intensity_counts = 0
    for token in review:
        if token[0]=='_' and token[1:] in lexicon_list: # not good = _good
            not_counts+=1
        elif token[-1]=='_' and token[1:] in lexicon_list: # really good = good_
            intensity_counts+=1
        elif token in lexicon_list:
            count+=1
    for word in capitals:
        if stemmer.stem(word) in lexicon_list:
            intensity_counts+=2
    data_row[name+'_count'] = count
    data_row[name+'_not_counts'] = not_counts
    data_row[name+'_intensity_count'] = intensity_counts
    return data_row

def count_lexicons_pos(data):
    data = data.apply(lambda row: count_num_lexicon(row,stemmed_POSITIVES,'pos'), axis=1)
    return data

def count_lexicons_neg(data):
    data = data.apply(lambda row: count_num_lexicon(row,stemmed_NEGATIVES,'neg'), axis=1)
    return data

def count_POS_tags_util(data_row):
    num_noun = 0
    num_adj = 0
    num_verb = 0
    num_adv = 0
    num_pn = 0
    for tag in data_row['pos_tags']:
        if tag=='NN' or tag=='NNS':
            num_noun+=1
        elif tag[:2]=='JJ':
            num_adj+=1
        elif tag[:2]=='VB':
            num_verb+=1
        elif tag[:2]=='RB':
            num_adv +=1
        elif tag=='PRP'or tag=='PRP$':
            num_pn+=1
    data_row['num_noun'] = num_noun
    data_row['num_adj'] = num_adj
    data_row['num_verb'] = num_verb
    data_row['num_adv'] = num_adv
    data_row['num_pn'] = num_pn
    return data_row

def count_POS_tags(data):
	data = data.apply(lambda row : count_POS_tags_util(row), axis=1)
	return data

def count_length(data):
    data['length'] = data['review'].apply(lambda x: len(x))
    return data

def get_senti_features_0(data, train, fast=True):
    if not fast: #-> _count, _not_counts, _intensity_counts
        data = run_parallel(data,count_lexicons_pos)
        data = run_parallel(data,count_lexicons_neg)
    else: #-> weighted_num_pos, weighted_num_neg ------> FASTER
        reviews = data['review'].apply(lambda x: " ".join(x))
        vectorizer_positive_lexicon = CountVectorizer(vocabulary=stemmed_POSITIVES,binary=False)
        vectorizer_negative_lexicon = CountVectorizer(vocabulary=stemmed_NEGATIVES,binary=False)
        if train:
            X_pos_lexicon = (vectorizer_positive_lexicon.fit_transform(reviews)).toarray().sum(axis=1)
            X_neg_lexicon = (vectorizer_negative_lexicon.fit_transform(reviews)).toarray().sum(axis=1)
        else:
            X_pos_lexicon = (vectorizer_positive_lexicon.transform(reviews)).toarray().sum(axis=1)
            X_neg_lexicon = (vectorizer_negative_lexicon.transform(reviews)).toarray().sum(axis=1)
        data['weighted_num_pos'] = X_pos_lexicon
        data['weighted_num_neg'] = X_neg_lexicon
    return data

def get_senti_features_1(data, train): # slow
    print("Calculating Sentiment Scores. . . ")
    vectorizer_positive_lexicon = CountVectorizer(vocabulary=stemmed_POSITIVES,binary=False)
    vectorizer_negative_lexicon = CountVectorizer(vocabulary=stemmed_NEGATIVES,binary=False)
    print('Forming matrix for Lexicon Dict counts')
    reviews = data['review'].apply(lambda x: " ".join(x))
    if train:
        X_pos_lexicon = (vectorizer_positive_lexicon.fit_transform(reviews)).toarray()
        X_neg_lexicon = (vectorizer_negative_lexicon.fit_transform(reviews)).toarray()
    else:
        X_pos_lexicon = (vectorizer_positive_lexicon.transform(reviews)).toarray()
        X_neg_lexicon = (vectorizer_negative_lexicon.transform(reviews)).toarray()
    X_pos_lexicon = np.matmul(X_pos_lexicon,pos_scores)
    X_neg_lexicon = np.matmul(X_neg_lexicon,neg_scores)
    data['polarity'] = X_pos_lexicon+X_neg_lexicon
    return data
    
def get_sentiments(data, train, get_scores=False):
    if get_scores:
        data = get_senti_features_1(data,train)
    else:
        data = get_senti_features_0(data,train,fast=fast)
    return data