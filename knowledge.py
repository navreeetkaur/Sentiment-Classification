from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
import re

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

POS_WORDS = ['cool', 'amaze', 'impress', 'rude', 'comfort', 'happy', 'great', 'good', 'excellent', 'beautiful',
           'best', 'awesome', 'cheap', 'recommend', 'suggest', 'fast', 'incredible', 'glad', 'yummi', 'tender', 
            'flavour','smile', 'polite', 'homemade', 'fabulous', 'refresh', 'best', 'heart', 'refreshing', 'great', ]

NEG_WORDS = ['hair', 'terrible', 'eww', 'yuck', 'worst', 'ok', 'okay', 'expensive', 'horrible', 'slow', 'lack',
            'average', 'unfortunate', 'smell', 'sad', 'bland', 'dirty', 'avoid', 'mess', 'weird', 'mediocre', 'annoy',
            'bad', 'digusting', 'bad', 'boring', 'dull', 'bore',]

CUSTOM_STOP = ['time', 'place', 'order', 'service', 'come', 'day', 'minute', 'hour', 'restraunt', 'food'
                            'came', 'went', 'people', 'know', 'menu', 'year', 'review', 'check', 'meal', 'boyfriend',
                            'dish', 'area', 'lunch', 'dinner', 'breakfast',  'guy', 'girl', 'man', 'friend',
                            'woman', 'item', 'husband', 'kid', 'buy', 'group', 'brought', 'ago', 'brunch', 
                            'girlfriend', 'daughter', 'son', 'mom', 'dad', 'sister', 'brother', 'afternoon', 'morning',
                            'night']


to_remove = ['too', 'just', 'ok']

stoplist = stopwords.words('english')+ list(ENGLISH_STOP_WORDS) + FOOD_WORDS + list(NEGATION) + list(INTENSITY) + CUSTOM_STOP

for word in to_remove:
  try:
    stoplist.remove(word)
  except:
    continue

STOPWORDS = set(stoplist)

stemmer = PorterStemmer()
# sentiment lexicons
opinionfinder = 'data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.txt'
# opinion lexicons: taken from https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
opinion_lexicon_pos = 'data/opinion-lexicon-English/positive-words.txt'
opinion_lexicon_neg = 'data/opinion-lexicon-English/negative-words.txt'
# general inquirer
general_inquirer = 'data/general-inquirer/inquirerbasicttabsclean.txt'

#- - - - - - - - - - Opinion Lexicons Bing Liu - - - - - - - - - - 
negs=[]
pos=[]
with open(opinion_lexicon_neg, 'rb') as f_neg: 
    neg_lines = f_neg.readlines()
    for i, line in enumerate(neg_lines):
        try:
            line = line.decode("utf-8").strip()
            if line and line[0]!=';':
                word = stemmer.stem(line)
                word = re.sub('[-]','_', word)
                if word not in negs:
                    negs.append(word)
        except UnicodeDecodeError:
            continue
with open(opinion_lexicon_pos, 'rb') as f_pos: 
    pos_lines = f_pos.readlines()
    for i, line in enumerate(pos_lines):
        try:
            line = line.decode("utf-8").strip()
            if line and line[0]!=';':
                word = stemmer.stem(line)
                word = re.sub('[-]','_', word)
                if word not in pos:
                    pos.append(word)
        except UnicodeDecodeError:
            continue

# - - - - - - - - - - General Inquirer - - - - - - - - - - 
# with open(general_inquirer, 'r') as f:
#     lines = f.readlines()
    
# for line in lines[1:]:
#     words = line.strip().split()
#     if ('Negativ' or 'Ngtv' in words):
#         lexicon = stemmer.stem(words[0].lower())
#         if lexicon not in negs:
#             negs.append(lexicon)
#     if ('Positiv' or 'Pstv' in words):
#         lexicon = stemmer.stem(words[0].lower())
#         if lexicon not in pos:
#             pos.append(lexicon)

# remove stopwords from positive and negative and stem
pos = pos+POS_WORDS
neg = negs+NEG_WORDS
POSITIVES = []
NEGATIVES = []
for word in set(pos):
    if len(word)>2 and word not in STOPWORDS:
        POSITIVES.append(stemmer.stem(word))
for word in set(negs):
    if len(word)>2 and word not in STOPWORDS:
        NEGATIVES.append(stemmer.stem(word))

LEXICON_DICT = POSITIVES + NEGATIVES




