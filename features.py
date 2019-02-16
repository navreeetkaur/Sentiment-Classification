import pickle
import numpy as np 
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer, PorterStemmer
from time import time 
from knowledge import STOPWORDS, POSITIVES, NEGATIVES

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR, NuSVC, NuSVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron, Lasso, Ridge
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, StratifiedKFold, StratifiedShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error

tqdm.pandas()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
pickle_train = 'data/df_train'
pickle_dev = 'data/df_dev'
df_train =  pd.read_pickle(pickle_train)
df_dev = pd.read_pickle(pickle_dev)


one = df_train_small.loc[df_train_small['rating'] == 1.0]
two = df_train_small.loc[df_train_small['rating'] == 2.0]
three = df_train_small.loc[df_train_small['rating'] == 3.0]
four = df_train_small.loc[df_train_small['rating'] == 4.0]
five = df_train_small.loc[df_train_small['rating'] == 5.0]

num_reviews = [one.shape[0],two.shape[0],three.shape[0],four.shape[0],five.shape[0]]
min_num_review = float(num_reviews.index(min(num_reviews))+1)


##################################  Length of review ################################## 
df_train['length'] = df_train['review'].progress_apply(lambda x: len(x))
df_dev['length'] = df_dev['review'].progress_apply(lambda x: len(x))

##################################  Sentiment Lexicons ################################## 

def count_num_lexicon(data_row, lexicon_list, name):
	review = data_row['review']
	capitals = data_row['capitals']
    count = 0
    not_counts = 0
    intensity_counts = 0
    for token in review:
    	if token[0]=='_' and token[1:] in lexicon_list: # not good = _good
        	not_counts+=1
        elif word[-1]=='_' and token[1:] in lexicon_list: # really good = good_
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


df_train = df_train.progress_apply(lambda row: count_num_lexicon(row,POSITIVES,'pos'), axis=1)
df_train = df_train.progress_apply(lambda row: count_num_lexicon(row,NEGATIVES,'neg'), axis=1)

df_dev = df_dev.progress_apply(lambda row: count_num_lexicon(row,POSITIVES,'pos'), axis=1)
df_dev = df_dev.progress_apply(lambda row: count_num_lexicon(row,NEGATIVES,'neg'), axis=1)

df_train.to_pickle('data/df_train_g0')
df_dev.to_pickle('data/df_dev_g0')


