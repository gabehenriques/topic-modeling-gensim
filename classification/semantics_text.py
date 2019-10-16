#!/usr/bin/env python3
# Copyright (C) 2019 Gabriel Henriques

import re
import pickle
import gensim
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import brown
from nltk.corpus import stopwords
from clean_text import PreProcessor

"""Topic Modeling;"""

data = []

"""
Prepare dataset;
The Brown Corpus was the first million-word electronic
corpus of English, created in 1961 at Brown University;
"""
for fileid in brown.fileids():
    document = ' '.join(brown.words(fileid))
    data.append(document)


"""This corpus contains text from 500 sources;"""
DATASET_SIZE = len(data)
NUM_TOPICS = 1000

"""Instantiate the preprocessor class"""
sentence_preprocessor = PreProcessor()


"""Tokenize dataset;"""
tokenized_data = []
for sentence in data:
    tokenized_data.append(sentence_preprocessor.preprocess_sentence(sentence))


'''Build a dictionary - associate words to numeric ids;'''
dictionary = corpora.Dictionary(tokenized_data)


"""Convert document into the bag-of-words (BoW);"""
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

"""
Implement Latent Dirichlet Allocation (LDA);
Classify text in a document to a particular topic;
"""
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
