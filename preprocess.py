import re
import json
# import string
# import operator
# from collections import OrderedDict
import numpy as np
import pandas as pd
import pickle

import multiprocessing as mp

import unidecode
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk.stem import LancasterStemmer, WordNetLemmatizer, SnowballStemmer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from knowledge import *
from knowledge import CONTRACTION_MAP, NEGATION, INTENSITY, STOPWORDS
from utils import run_parallel
from variables import *


##############################################  CLEANING ############################################## 

def clean_util(review):
    review = unidecode.unidecode(review) # convert è to e
    num_exclamations = len(re.findall('!+', review)) # number of exclamations
    review = re.sub('[\r\t\n\f]',' ', review) 
    review = re.sub('[0-9]','', review) # remove numbers
    # remove punctuations
    review = re.sub('[$|&|%|@|(|*|)|~|:|;|\-|^|?|!|,|`|_|/|\\\\|<|>|#|&|+|=|{|}|..+]', ' ', review)
    review = re.sub('[\'|\"]', '', review)
    # replace "goooood" with "goood" OR good_ 
    review = re.sub(r'(.)\1{2,}', r'\1\1\1',review)
    # find all capital words
    cap_words = list(map(lambda x:x.lower(),re.findall('[A-Z][A-Z][A-Z]+', review) + re.findall('\*(.*?)\*', review)))
    review = review.lower()
    # expand contractions
    words = review.split()
    review = " ".join([CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in words])

    return review, cap_words, num_exclamations

def clean(data):
    data['review'], data['capitals'], data['num_ex'] = zip(*data['review'].apply(lambda x: clean_util(x)))
    return data

##############################################  POS TAGGING ########################################### 
def tag_with_pos_util(review):
    pos = []
    for word in review:
        pos.append(pos_tag([word])[0][1].upper())
    return pos

def tag_with_pos(data):
    data['pos'] = data['review'].apply(lambda word: tag_with_pos_util(word_tokenize(word)))
    return data

##############################################  LEMMATIZATION #########################################
def lemma_util(data_row):
    tokens = word_tokenize(data_row['review'])
    words = []
    append_neg = False
    append_intensity = False
    for i, token in enumerate(tokens):
        if token in NEGATION:
            append_neg=True
            continue
        if token in INTENSITY:
            append_intensity= True
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
            
        if append_neg: 
            words.append('_'+lemmatizer.lemmatize(token, pos=tag_pos(data_row['pos'][i])))
            append_neg=False
        elif append_intensity:
            words.append(lemmatizer.lemmatize(token, pos=tag_pos(data_row['pos'][i]))+'_')
            append_intensity=False
        else:
            words.append(lemmatizer.lemmatize(token, pos=tag_pos(data_row['pos'][i])))
    data_row['review'] = words
    return data_row

def lemmatize(data):
    data = data.apply(lambda row: lemma_util(row), axis=1)
    return data

##############################################  STEMMING ############################################## 

def stem_util(data_row):
    tokens = word_tokenize(data_row['review'])
    words = []
    pos = []
    append_neg = False
    append_intensity = False
    
    for i, token in enumerate(tokens):
        if token in NEGATION:
            append_neg=True
            continue
        if token in INTENSITY:
            append_intensity= True
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
            
        if append_neg: 
            if TAG:
                pos.append(pos_tag([token])[0][1])
            words.append('_'+stemmer.stem(token))
            append_neg=False
        elif append_intensity:
            if TAG:
                pos.append(pos_tag([token])[0][1])
            words.append(stemmer.stem(token)+'_')
            append_intensity=False
        else:
            if TAG:
                pos.append(pos_tag([token])[0][1])
            words.append(stemmer.stem(token))
            
    data_row['review'] = words
    if TAG:
        data_row['pos_tags'] = pos
    
    return data_row

def stem(data):
    data = data.apply(lambda row: stem_util(row), axis=1)
    return data

def get_dataframe(FILE,train,save=False,pickle_file=None):
    if train:
        review = []
        ratings = []
        if verbose:
            print("Reading data. . .")
        i=0
        for line in (open(FILE, 'r')):
            # if i>=200000:
            #     break
            line = json.loads(line)
            review.append(line['review'])
            ratings.append(line['ratings'])
            i+=1
        df = pd.DataFrame({"rating":ratings, "review":review})
        del review
        del ratings
        if verbose: 
            print("Cleaning data. . .")
        df = clean(df)
        if verbose:
            print("Stemming data. . .")
        df = run_parallel(df, stem, save=save, pickle_file=pickle_file)
    else:
        review = []
        if verbose:
            print("Reading data. . .")
        i=0
        for line in (open(FILE, 'r')):
            # if i>=50000:
            #     break
            line = json.loads(line)
            review.append(line['review'])
            i+=1
        df = pd.DataFrame({"review":review})
        del review
        if verbose: 
            print("Cleaning data. . .")
        df = clean(df)
        if verbose:
            print("Stemming data. . .")
        df = run_parallel(df, stem, save=save, pickle_file=pickle_file)
    return df



