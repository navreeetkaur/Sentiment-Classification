from nltk.stem import WordNetLemmatizer, PorterStemmer

stack_features = True

TAG=False # POS tag
remove_stopwords=True
verbose=True
get_scores=True # for sentiment lexicon scores - calculate polarity scores
fast=True # for sentiment lexicon scores - use count vectorizer for fast 
use_svr = True
form_vocab=False # form custom vocabulary
vocab_size=10000 # size of custom vocabulary
use_tfidf = True

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

if use_tfidf:
	params = {'binary':False,'max_features':10000,'ngram_range':(1, 2)}#,'max_df':1,'min_df':0.01,, 'norm':'l2'}
else:
	params = {'binary':False,'max_features':10000,'ngram_range':(1, 2)}#,'max_df':1,'min_df':0.01,}
not_features = ['capitals','review', 'rating', 'pos_tags','pos_polarity','neg_polarity','num_noun','num_adj','num_verb','num_adv', 'num_pn']
# not_features = ['capitals','review', 'rating', 'pos_tags', 'pos_polarity','neg_polarity','length', 'num_noun','num_adj','num_verb','num_adv', 'num_pn']

stemmed_pos_file = 'data/stemmed_pos_file.pkl'
stemmed_neg_file = 'data/stemmed_neg_file.pkl'
pickle_vocab_vector = 'data/vocab.pkl'