import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import FastText, Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models.callbacks import ConvergenceMetric
import numpy as np
import pandas as pd

from pathlib import Path
from scipy import spatial

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

np.random.seed(1001)

_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def get_sent_tokenized_lemmas(b):
    return [[normalize_word(t) for t in nltk.word_tokenize(s)] for s in nltk.sent_tokenize(b)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    print("Genreating or loading")
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def fast():
    # sentencesa = [s.strip() for s in data]
    # sentencesa = [word_tokenize(s) for s in sentencesa]
    model_path = 'features/w3c.pkl'
    epochs = 15
    size = 50
    workers = 6
    if not Path(model_path + 'head').exists() or not Path(model_path + 'body').exists():
        stances = pd.read_csv('fnc-1/train_stances.csv').values.tolist()
        bodies = dict(pd.read_csv('fnc-1/train_bodies.csv').values.tolist())
        clean_heads = []
        clean_bodies = []
        for headline, body_id, stance in tqdm(stances):

            body = bodies[body_id]
            clean_headline = clean(headline)
            clean_body = clean(body)
            clean_headline = get_tokenized_lemmas(clean_headline)
            clean_body = get_sent_tokenized_lemmas(clean_body)

            clean_heads.append(clean_headline)
            clean_bodies.extend(clean_body)

        print(f"Training embedding model on {len(clean_heads)} sentences for {epochs} epochs")
        model_h = FastText(sentences=clean_heads, size=size, iter=epochs, sg=1, workers=workers)
        print("Saving")
        model_h.save(model_path + 'head')
        print("Saving done")

        print(f"Training embedding model on {len(clean_heads)} sentences for {epochs} epochs")
        model_b = FastText(sentences=clean_bodies, size=size, iter=epochs, sg=1, workers=workers)
        print("Saving")
        model_b.save(model_path + 'body')
        print("Saving done")

    else:
        print("loading embedding")
        model_h = FastText.load(model_path + 'head')
        model_b = FastText.load(model_path + 'body')
        print("loading done")
    return model_h, model_b


def embedding_features(headlines, bodies):
    print("embedding_features:")
    clean_headlines = []
    clean_bodies = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = " ".join(get_tokenized_lemmas(clean_headline))
        body_s = get_sent_tokenized_lemmas(clean_body)
        clean_headlines.append(clean_headline)
        clean_bodies.append(body_s)

    X = []
    model_h, model_b = fast()
    for i, (clean_h, clean_b) in tqdm(enumerate(zip(clean_headlines, clean_bodies))):
        body_emb = np.zeros(50)
        for s in clean_b:
            body_emb = body_emb + model_b[" ".join(s)]
        body_emb = body_emb / len(clean_b)

        h = [float(x) for x in model_h[clean_h].tolist()]
        b = [float(x) for x in body_emb.tolist()]
        sim = 1.0 - spatial.distance.cosine(h, b)

        X.append([sim] + h + b)
    return X


def word_overlap_features(headlines, bodies):
    print("word_overlap_features:")
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    print("refuting_features")
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    print("polarity_features")
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X
