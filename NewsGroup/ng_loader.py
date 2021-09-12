import os
import sys
import numpy as np
import torch

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def ng_loader():
    BASE_DIR = 'NewsGroup'
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2


    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-(num_validation_samples+900)]                                  #Train set
    y_train = labels[:-(num_validation_samples+900)]
    x_pval = data[data.shape[0]-num_validation_samples-900:(data.shape[0]           #Validation set
                                                        -num_validation_samples)]
    y_pval = labels[data.shape[0]-(num_validation_samples+900):(data.shape[0]
                                                        -num_validation_samples)]
    x_val = data[-num_validation_samples:]                                          #Test set
    y_val = labels[-num_validation_samples:]

    print (data.shape[0] - num_validation_samples)
    print ('XPVAL: ', x_pval.shape, data.shape)

    print('Preparing embedding matrix.', x_train.shape)

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = torch.zeros(num_words, EMBEDDING_DIM)
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = torch.from_numpy(embedding_vector)

    return embedding_matrix, x_train, y_train, x_pval, y_pval, x_val, y_val, num_words, EMBEDDING_DIM