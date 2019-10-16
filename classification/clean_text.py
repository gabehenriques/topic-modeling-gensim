#!/usr/bin/env python3
# Copyright (C) 2019 Gabriel Henriques

import re
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''Text Preprocessor;'''

emotions = set([
      ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
      ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
      ':c', ':{', '>:\\', ';(', '...', ':d', '>:d', '-', '--', ':-)', ':)', ';)',
      ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D',
      '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)",
      ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p',
      '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'
      ])

class PreProcessor:

    '''Clean, tokenize and parse sentences;'''

    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + list(emotions))

    def preprocess_sentence(self, sentence):

        sentence_tokens = word_tokenize(self._preprocess_sentence(sentence))
        preprocessed_sentences=[]
        stemmer = PorterStemmer()

        for word in sentence_tokens:
            if word not in self._stopwords:
                stem_word = stemmer.stem(word)
                preprocessed_sentences.append(stem_word)
        return preprocessed_sentences

    def _preprocess_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'\$\w*', '', sentence)
        sentence = re.sub(r'^RT[\s]+', '', sentence)
        sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
        sentence = re.sub(r'#', ' ', sentence)
        sentence = re.compile("["
            U"\U00002600-\U000026FF"
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # dingbats
                               "]+", flags=re.UNICODE).sub(r'', sentence)
        return sentence
