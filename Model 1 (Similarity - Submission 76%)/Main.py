from utils.dataset import DataSet
from fnc_kfold import kfold_split
from fnc_kfold import get_stances_for_folds
from fnc_kfold import GradientBoostingClassifier, check_version, parse_params
from fnc_kfold import generate_features
from utils.score import score_submission, print_confusion_matrix, report_score, LABELS
import numpy as np
import pandas as pd
import os
import re
from nltk import word_tokenize
from nltk import FreqDist
import numpy
import gensim
dataset = DataSet()

check_version()
parse_params()

#Load the training dataset and generate folds
folds,hold_out = kfold_split(dataset,n_folds=10)#10
fold_stances, hold_out_stances = get_stances_for_folds(dataset,folds,hold_out)

# Load the competition dataset
competition_dataset = DataSet("competition_test")
X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

Xs = dict()
ys = dict()

# Load/Precompute all features now
X_holdout,y_holdout = generate_features(hold_out_stances,dataset,"holdout")
for fold in fold_stances:
    Xs[fold],ys[fold] = generate_features(fold_stances[fold],dataset,str(fold))

best_score = 0
best_fold = None

print('*' * 80)
print(os.getcwd())
print('*' * 80)
print('*' * 80)
print('*' * 80)
print('*' * 80)
print('*' * 80)
print('*' * 80)
print('*' * 80)

#Inputting Data for Word2Vec
train_heads = pd.read_csv("fnc-1/train_stances.csv")
train_bodies = pd.read_csv("fnc-1/train_bodies.csv")
competition_heads = pd.read_csv("fnc-1/competition_test_stances.csv")
competition_bodies = pd.read_csv("fnc-1/competition_test_bodies.csv")
print(train_heads.shape)
print(train_bodies.shape)

#Preprocessing data
j = open("stopwords.txt",'r')
stop_w = j.read().split('\n')
j.close()
def preprocesser(data, col):
    table = []
    table = [re.sub("[^a-zA-Z]", " ",str(a)) for a in data[col]]
    table = [word_tokenize(a) for a in table]
    table = [[w.lower() for w in a] for a in table]
    table = [[w for w in a if w not in stop_w] for a in table]
    table = [' '.join(x) for x in table]
    data[col] = table

def merger(data1, data2, left_on=['Body ID'], right_on=['Body ID']):
    full_data = pd.merge(data1, data2,  how='inner', left_on=['Body ID'], right_on=['Body ID'])
    return full_data

def get_unique_strings(dataframe, colname1='Headline', colname2='articleBody'):
    unique_strings = []
    for index, row in dataframe.iterrows():
        if row[colname1] not in unique_strings:
            unique_strings.append(row[colname1])
        if row[colname2] not in unique_strings:
            unique_strings.append(row[colname2])
    return unique_strings

def unique_counter(list_of_text):
    tokens = []
    for i in list_of_text:
        tokens.extend(word_tokenize(i))
    freq = FreqDist(tokens)
    number_of_words = len(freq)
    return number_of_words          

def word2vecmodel(data):
    
    from gensim.test.utils import common_texts, get_tmpfile
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize

    numpy.random.seed(1001)

    sentences = [s.strip() for s in data]
    numpy.random.shuffle(sentences)
    sentences = [word_tokenize(s) for s in sentences]
    
    model = gensim.models.Word2Vec(sentences, size=300, window=2, min_count=1, sg=1, negative = 10,iter=10)
    w1="good"
    a = model.wv.most_similar (positive=w1, topn=20)
    print(a)
    model.save('w2c.pkl')
   
    return model;
def fast(data):
    
    from gensim.test.utils import common_texts, get_tmpfile
    from gensim.models import FastText
    from nltk.tokenize import word_tokenize

    numpy.random.seed(1001)

    sentencesa = [s.strip() for s in data]
    numpy.random.shuffle(sentencesa)
    sentencesa = [word_tokenize(s) for s in sentencesa]
    
    model = gensim.models.FastText(size=4, window=3, min_count=1, sentences=sentencesa, iter=10)

    model.save('w3c.pkl')
   
    return model;

preprocesser(train_heads, 'Headline')
preprocesser(train_bodies, 'articleBody')

preprocesser(competition_heads, 'Headline')
preprocesser(competition_bodies, 'articleBody')

train = merger(train_heads, train_bodies)
competition = merger(competition_heads, competition_bodies)

print(train.shape)
print(train)
unique_trains = get_unique_strings(train)#train

total_number_of_words = unique_counter(unique_trains)
print(total_number_of_words)
total_number_of_words

df = pd.DataFrame(train)
df['sentence']=df['Headline'].str.cat(df['articleBody'],sep=" ")
print(df['sentence'])
#embedding = fast(unique_trains)
embedding = word2vecmodel(unique_trains) #df['sentence']
print(embedding)

from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Dense, Input, Dropout, SpatialDropout1D, LSTM, Embedding, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence, Tokenizer
from numpy import array
from collections import Counter
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.optimizers import SGD

#Hyperparameters
MAX_SENT_LEN = 145 #145
MAX_VOCAB_SIZE = len(unique_trains)
EMBEDDING_DIM = 300
BATCH_SIZE = 32
N_EPOCHS = 20
LSTM_DIM = 128

# Classifier for each fold
for fold in fold_stances:
    ids = list(range(len(folds)))
    del ids[fold]
    
    X_train = np.vstack(tuple([Xs[i] for i in ids])) 
    y_train = np.hstack(tuple([ys[i] for i in ids]))
         
    print(X_train.shape)
    print(X_train)
    print(y_train.shape)
    print("&" * 80)
    print(len(ids))
    print(len(folds))
    print("&" * 80)
    X_test = Xs[fold]
    y_test = ys[fold]

    #FastText emb -> Relu -> Linear Layer -> softmax -> (agree, disagree, discuss, unrelated)
    model1 = Sequential()

    #e = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim = EMBEDDING_DIM, weights=[embeddings_matrix],input_length = MAX_SENT_LEN, trainable=False, name='word_embedding_layer')

    #model1.add(e)

    #model1.add(LSTM(128,dropout = 0.2, recurrent_dropout=0.3))

    #model1.add(Dense(200,kernel_regularizer=regularizers.l2(0.01),name = 'hidden_layer'))#0.01

    #model1.add(Dropout(0.2))#0.1, 0.2

    #model1.add(Flatten())
    
    model1.add(Dense(70,activation = 'relu', input_shape = (145, )))
    model1.add(Dropout(0.7))
    model1.add(Dense(70,kernel_regularizer=regularizers.l2(0.01),name = 'hidden_layer'))#0.01
    model1.add(Activation(activation = 'relu', name = 'activation_1')) #model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.7))

    model1.add(Dense(4, activation='softmax', name = 'output_layer'))

    model1.summary()
    #sparse_categorical_crossentropy
    #adam
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    model1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test))
    
    predicted = [LABELS[numpy.argmax(a)] for a in model1.predict(X_test)]
    print(model1.predict(X_test).shape)
    actual = [LABELS[int(a)] for a in y_test]

    fold_score, _ = score_submission(actual, predicted)
    max_fold_score, _ = score_submission(actual, actual)

    score = fold_score/max_fold_score

    print("Score for fold "+ str(fold) + " was - " + str(score))
    if score > best_score:
        best_score = score
        best_fold = model1

#Run on Holdout set and report the final score on the holdout set
predicted = [LABELS[int(numpy.argmax(a))] for a in best_fold.predict(X_holdout)]
actual = [LABELS[int(a)] for a in y_holdout]

print("Scores on the dev set")
report_score(actual,predicted)


#Run on competition dataset
predicted = [LABELS[numpy.argmax(a)] for a in best_fold.predict(X_competition)] #best_fold
actual = [LABELS[int(a)] for a in y_competition]

print("Scores on the test set")
report_score(actual,predicted)

bodyid=[]
headlines=[]
predictions=[]
actuals=[]

for a in competition_dataset.stances:
    bodyid.append(a['Body ID'])
    headlines.append(a['Headline'])
for a in predicted:
    predictions.append(a)
for a in actual:
    actuals.append(a)

import pandas as pd

df = pd.DataFrame({"Headline" : headlines, "Body ID" : bodyid,"Stance": predictions})
df.to_csv("submission.csv", index=False)

#np.savetxt('Submission_JA.csv',np.c_[headlines, bodyid, predictions],header="Headline, Body ID, Stance", fmt = '%s')

#np.savetxt('body_id.csv', bodyid, delimiter='/n', fmt='%i')
#np.savetxt('stance.csv', predictions, delimiter='/n', fmt='%s')
#np.savetxt('headline.csv', headlines, delimiter='/n', fmt='%s')



