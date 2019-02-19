from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import nltk
from time import time
from utils import penn_to_wn
from utils import *
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

"""
Expanding contraction - with minor changes, [SOURCE]: https://www.kaggle.com/saxinou/nlp-01-preprocessing-data
"""
CONTRACTION_MAP = {"aint": "is not", "arent": "are not","cant": "cannot", "tis": "it is", "twas": "it was",
                   "cantve": "cannot have", "'cause": "because", "couldve": "could have", 
                   "couldnt": "could not", "couldntve": "could not have","didnt": "did not", 
                   "doesnt": "does not", "dont": "do not", "hadnt": "had not", 
                   "hadntve": "had not have", "hasnt": "has not", "havent": "have not", 
                   "hed": "he would", "hedve": "he would have", "hell": "he will", 
                   "hellve": "he he will have", "hes": "he is", "howd": "how did", 
                   "howdy": "how do you", "how'll": "how will", "how's": "how is", 
                   "id": "i would", "idve": "i would have", "i'll": "i will", 
                   "illve": "i will have","im": "i am", "ive": "i have", 
                   "isnt": "is not", "itd": "it would", "itdve": "it would have", 
                   "itll": "it will", "itllve": "it will have","its": "it is", 
                   "lets": "let us", "maam": "madam", "maynt": "may not", 
                   "mightve": "might have","mightnt": "might not","mightntve": "might not have", 
                   "mustve": "must have", "mustnt": "must not", "mustntve": "must not have", 
                   "neednt": "need not", "needntve": "need not have","oclock": "of the clock", 
                   "oughtnt": "ought not", "oughtntve": "ought not have", "shant": "shall not",
                   "shant": "shall not", "shantve": "shall not have", 
                   "shed": "she would", "she's": "she is", "we're": "we are", "weve": "we have",
                   "shedve": "she would have", "shell": "she will", "shellve": "she will have", 
                   "shouldve": "should have", "shouldnt": "should not", 
                   "shouldntve": "should not have", 
                   "sove": "so have","sos": "so as", "wedve": "we would have", "well": "we will", 
                   "thiss": "this is","thatd": "that would", "thatdve": "that would have","thats": "that is", 
                   "thered": "there would", "theredve": "there would have","theres": "there is", 
                   "theyd": "they would", "theydve": "they would have", "theyll": "they will", 
                   "theyllve": "they will have", "theyre": "they are", "theyve": "they have", 
                    "wasnt": "was not", "wed": "we would", "tove": "to have", "wellve": "we will have",
                    "werent": "were not", "whatll": "what will", "whatllve": "what will have", "whatre": "what are", 
                   "whats": "what is", "whatve": "what have", "whens": "when is", 
                   "whenve": "when have", "whered": "where did", "wheres": "where is", 
                   "whereve": "where have", "wholl": "who will", "whollve": "who will have", 
                   "whos": "who is", "whove": "who have", "whys": "why is", 
                   "whyve": "why have", "willve": "will have", "wont": "will not", 
                   "wontve": "will not have", "wouldve": "would have", "wouldnt": "would not", 
                   "wouldntve": "would not have", "yall": "you all", "yalld": "you all would",
                   "yalldve": "you all would have","yallre": "you all are","yallve": "you all have",
                   "youd": "you would", "youdve": "you would have", "youll": "you will", 
                   "youllve": "you will have", "youre": "you are", "youve": "you have" } 

NEGATION = set(["not", "no",  "neither", "except", "never"])

INTENSITY = set(["mostly", "really", "very", "extremely"])

FOOD_WORDS = ['food','burger', 'pizza', 'sandwish', 'meat', 'coffee','sushi', 'taco', 'beef', 'salad','thai', 'pie',
             'steak', 'soup', 'dessert', 'egg', 'fish', 'pork', 'dog', 'potato', 'wine', 'cake', 'chips', 'noodles', 
             'onion', 'bacon', 'bbq', 'pasta', 'burrito', 'italian', 'waffle', 'mushroom', 'corn', 'sausage', 'avocado' 
             'milk', 'fruit', 'donut', 'juice', 'margarita', 'lemon', 'indian', 'korean', 'desert', 'lettuce', 'snack'
             'japanese', 'oyster', 'nacho', 'tofu', 'cheese', 'banana', 'apple', 'strawberry', 'mango', 'orange', 
              'honey', 'shake', 'meatball', 'olive',  ]

POS_WORDS = ['cool', 'amaze', 'impress', 'comfort', 'happy', 'great', 'good', 'excellent', 'beautiful', 'like', 'nice', 'well'
           'best', 'awesome', 'cheap', 'recommend', 'suggest', 'fast', 'incredible', 'glad', 'yummi', 'yummy', 'tender', 
            'flavour','smile', 'polite', 'homemade', 'fabulous', 'refresh', 'best', 'heart', 'refreshing', 'great', ]

NEG_WORDS = ['hair', 'terrible', 'eww', 'yuck', 'rude', 'worst', 'ok', 'okay', 'expensive', 'horrible', 'slow', 'lack',
            'average', 'unfortunate', 'smell', 'sad', 'bland', 'dirty', 'avoid', 'mess', 'weird', 'mediocre', 'annoy',
            'bad', 'digusting', 'bad', 'boring', 'dull', 'bore', 'pissed']

CUSTOM_STOP = ['time', 'place', 'order', 'service', 'come', 'day', 'minute', 'hour', 'restraunt', 'food'
                            'came', 'went', 'people', 'know', 'menu', 'year', 'review', 'check', 'meal', 'boyfriend',
                            'dish', 'area', 'lunch', 'dinner', 'breakfast',  'guy', 'girl', 'man', 'friend',
                            'woman', 'item', 'husband', 'kid', 'buy', 'group', 'brought', 'ago', 'brunch', 
                            'girlfriend', 'daughter', 'son', 'mom', 'dad', 'sister', 'brother', 'afternoon', 'morning',
                            'night']


to_remove = ['too', 'just', 'ok', 'well'] + POS_WORDS + NEG_WORDS

stoplist = stopwords.words('english')+ list(ENGLISH_STOP_WORDS) + FOOD_WORDS + list(NEGATION) + list(INTENSITY) + CUSTOM_STOP

for word in to_remove:
  try:
    stoplist.remove(word)
  except:
    continue

STOPWORDS = set(stoplist)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# opinion lexicons: taken from https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
opinion_lexicon_pos = 'opinion-lexicon-English/positive-words.txt'
opinion_lexicon_neg = 'opinion-lexicon-English/negative-words.txt'

def get_pos_and_neg_list():
    #- - - - - - - - - - Opinion Lexicons Bing Liu - - - - - - - - - - 
    print("Reading Opinion Lexicons . . . ")
    negs=[]
    with open(opinion_lexicon_neg, 'rb') as f_neg: 
        neg_lines = f_neg.readlines()
        for i, line in enumerate(neg_lines):
            try:
                line = line.decode("utf-8").strip()
                if line and line[0]!=';':
                    word = line
                    # word = stemmer.stem(line)
                    # word = re.sub('[-]','_', word)
                    if word not in negs:
                        negs.append(word)
            except UnicodeDecodeError:
                continue
    pos=[]
    with open(opinion_lexicon_pos, 'rb') as f_pos: 
        pos_lines = f_pos.readlines()
        for i, line in enumerate(pos_lines):
            try:
                line = line.decode("utf-8").strip()
                if line and line[0]!=';':
                    word=line
                    # word = stemmer.stem(line)
                    # word = re.sub('[-]','_', word)
                    if word not in pos:
                        pos.append(word)
            except UnicodeDecodeError:
                continue

    # remove stopwords from positive and negative and stem
    POSITIVES = []
    NEGATIVES = []
    pos = pos + POS_WORDS
    negs = negs + NEG_WORDS

    for word in set(pos):
        if len(word)>2 and '-' not in word and word not in STOPWORDS:
            POSITIVES.append(word)
    for word in set(negs):
        if len(word)>2 and '-' not in word and word not in STOPWORDS:
            NEGATIVES.append(word)


    return POSITIVES, NEGATIVES

def get_lexicon_dictionary(get_scores = True):
    POSITIVES, NEGATIVES = get_pos_and_neg_list()
    stemmed_POSITIVES=[]
    stemmed_NEGATIVES=[]
    pos_scores = []
    neg_scores = []
    pos_scores_custom = []
    neg_scores_custom = []

    for word in POSITIVES:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_POSITIVES or stemmed_word in stemmed_NEGATIVES:
            continue
        else:
            stemmed_POSITIVES.append(stemmed_word)
        if not get_scores:
            continue
        pos_tag = nltk.pos_tag([word])[0][1]
        wn_tag = penn_to_wn(pos_tag)      
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            pos_scores.append(0.2)
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            pos_scores.append(0.2)
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            pos_scores.append(0.2)
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        score = swn_synset.pos_score()-swn_synset.neg_score()
        if score==0:
            score=0.2
        pos_scores.append(score)


    for word in NEGATIVES:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_NEGATIVES or stemmed_word in stemmed_POSITIVES:
            continue
        else:
            stemmed_NEGATIVES.append(stemmed_word)
        if not get_scores:
            continue
        pos_tag = nltk.pos_tag([word])[0][1]
        wn_tag = penn_to_wn(pos_tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            neg_scores.append(-0.2)
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            neg_scores.append(-0.2)
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            neg_scores.append(-0.2)
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        score = swn_synset.pos_score()-swn_synset.neg_score()
        if score==0:
            score=-0.2
        neg_scores.append(score)


    for word in POS_WORDS:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_POSITIVES or stemmed_word in stemmed_NEGATIVES:
            continue
        else:
            stemmed_POSITIVES.append(stemmed_word)
        if not get_scores:
            continue
        pos_tag = nltk.pos_tag([word])[0][1]
        wn_tag = penn_to_wn(pos_tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            pos_scores_custom.append(0.2)
            continue
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            pos_scores_custom.append(0.2)
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            pos_scores_custom.append(0.2)
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        score = swn_synset.pos_score()-swn_synset.neg_score()
        if score<=0:
            score=0.2
        pos_scores_custom.append(score)

    for word in NEG_WORDS:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in stemmed_NEGATIVES or stemmed_word in stemmed_POSITIVES:
            continue
        else:
            stemmed_NEGATIVES.append(stemmed_word)
        if not get_scores:
            continue
        pos_tag = nltk.pos_tag([word])[0][1]
        wn_tag = penn_to_wn(pos_tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            neg_scores_custom.append(-0.2)
            continue       
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            neg_scores_custom.append(-0.2)
            continue  
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            neg_scores_custom.append(-0.2)
            continue
        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        score = swn_synset.pos_score()-swn_synset.neg_score()
        if score>=0:
            score=-0.2
        neg_scores_custom.append(score)

    if get_scores:
        pos_scores = pos_scores + pos_scores_custom
        neg_scores = neg_scores + neg_scores_custom

    return stemmed_POSITIVES, stemmed_NEGATIVES, pos_scores, neg_scores 



