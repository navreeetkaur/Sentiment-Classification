import re
import json
import string
import operator
from collections import OrderedDict
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import pickle

import multiprocessing as mp

import unidecode
from autocorrect import spell
import contractions
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.stem import LancasterStemmer, WordNetLemmatizer, SnowballStemmer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from knowledge import CONTRACTION_MAP, NEGATION, INTENSITY, STOPWORDS
##############################  VARIABLES ###########################################  
FILE_DEV = 'data/dev.json'
FILE_TRAIN = 'data/train.json'
pickle_clean_train = 'data/df_clean_train'
pickle_clean_dev = 'data/df_clean_dev'
pickle_train = 'data/df_train'
pickle_dev = 'data/df_dev'

lemmatizer = WordNetLemmatizer()
# stemmer = LancasterStemmer()
stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english')

num_partitions=100
num_cores=5

tqdm.pandas()

dev=[]
review_train = []
ratings_train = []
review_dev = []
ratings_dev = []

step00 = time()

print("STEP 0 : Reading dev data. . .")
for line in tqdm(open(FILE_DEV, 'r')):
    line = json.loads(line)
    review_dev.append(line['review'])
    ratings_dev.append(line['ratings'])

print("STEP 0 : Reading training data. . .")
for line in tqdm(open(FILE_TRAIN, 'r')):
    line = json.loads(line)
    review_train.append(line['review'])
    ratings_train.append(line['ratings'])   


df_train = pd.DataFrame({"rating":ratings_train, "review":review_train})
df_dev = pd.DataFrame({"rating":ratings_dev, "review":review_dev})

del review_train
del ratings_train
del review_dev
del ratings_dev

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
    data['review'], data['capitals'], data['num_ex'] = zip(*data['review'].progress_apply(lambda x: clean_util(x)))
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
        if token in STOPWORDS:
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
    data = data.progress_apply(lambda row: lemma_util(row), axis=1)
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
        if token in STOPWORDS:
            continue
            
        if append_neg: 
            pos.append(pos_tag([token])[0][1])
            words.append('_'+stemmer.stem(token))
            append_neg=False
        elif append_intensity:
            pos.append(pos_tag([token])[0][1])
            words.append(stemmer.stem(token)+'_')
            append_intensity=False
        else:
            pos.append(pos_tag([token])[0][1])
            words.append(stemmer.stem(token))
            
    data_row['review'] = words
    data_row['pos_tags'] = pos
    
    return data_row

def stem(data):
    data = data.progress_apply(lambda row: stem_util(row), axis=1)
    return data

##############################################  MULTIPROCESSING ########################################
def pos_and_lemma(data):
    data = lemmatize(tag_with_pos(data))
    return data

def pos_and_stem(data):
    data = stem(tag_with_pos(data))
    return data

def preprocess(data, pickle_file, method, num_tasks=100, num_cores=4):
    pool = mp.Pool(num_cores)
    # pbar = tqdm(total=len(num_tasks))
    # for i, _ in tqdm(enumerate(pool.imap_unordered(method, range(0, num_tasks)))):
    #             pbar.update()
    data = np.array_split(data, num_partitions)
    data = pd.concat(tqdm(pool.map(method, data),total=num_tasks))
    pool.close()
    pool.join()
    # pbar.close()
    data.to_pickle(pickle_file)
    return data

step0 = time()
print(f'time: {format((step0-step00)/60,'.0f')}:{format((step0-step00)%60,'.0f')}')
print("STEP 1.1: Cleaning training data. . .")
df_train = clean(df_train)
print("STEP 1.2: Cleaning dev data. . .")
df_dev = clean(df_dev)

step1 = time()
print(f'time: {(step1-step0)/60}:{(step1-step0)%60}')
print("STEP 2.1: Stemming training data. . .")
df_train = preprocess(df_train, pickle_train, stem)
print("STEP 2.2: Stemming dev data. . .")
df_dev = preprocess(df_dev, pickle_dev, stem)

step2 = time()
print(f'time: {(step2-step1)/60}:{(step2-step1)%60}')






