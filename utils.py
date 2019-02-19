import pandas as pd
import numpy as np
import pickle
import multiprocessing as mp
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from collections import OrderedDict

num_tasks=100
num_cores=4

def run_parallel(data, method, num_tasks=100, num_cores=4, save=False,pickle_file=None):
    pool = mp.Pool(num_cores)
    # pbar = tqdm(total=len(num_tasks))
    # for i, _ in tqdm(enumerate(pool.imap_unordered(method, range(0, num_tasks)))):
    #             pbar.update()
    data = np.array_split(data, num_tasks)
    data = pd.concat(pool.map(method, data))
    pool.close()
    pool.join()
    # pbar.close()
    if save:
        data.to_pickle(pickle_file)
    return data


def get_counter(reviews):
    counter = {}
    for review in (reviews):
        for word in review:
            counter[word] = counter.get(word,0) + 1
    return sorted(counter.items(), key=lambda pair:pair[1], reverse=True)


def build_vocab_size(counter, vocab_size):
    vocab = OrderedDict()
    for word, freq in counter:
        if len(vocab)>=vocab_size:
            return vocab
        vocab[word] = len(vocab)
    return vocab


def build_vocab_threshold(counter, threshold):
    vocab = OrderedDict()
    for word, freq in counter:
        if freq>threshold:
            vocab[word] = len(vocab)
    return vocab


def build_vocab_uniform(fds, num_per_class, threshold):
    # fds - counter per class
    vocab = OrderedDict()
    j = 0
    for counter in (fds):
        print(j)
        i=0
        for word, freq in counter: 
            if i>num_per_class:
                break
            elif word in vocab:
                #i+=1
                continue
            elif freq>threshold:
                vocab[word] = len(vocab)
            i+=1
        j+=1
    return vocab


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def reg_metrics(y_test, y_pred):
    evs = explained_variance_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    #mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #msle = mean_squared_log_error(y_test, y_pred)
    #meae = median_absolute_error(y_test, y_pred)
    print(f'evs:{evs}\nmse:{mse}\nr2:{r2}')

def visualizeConfusion(X):
    ax = sns.heatmap(X, annot=True, fmt="d", cbar = False)
    plt.title('Confusion matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

def class_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    avg_prec = average_precision_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f'acc:{acc}\naverage precion:{avg_prec}\nf1:{f1}')
    visualizeConfusion(conf_mat)