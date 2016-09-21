#!/usr/bin/env python
# coding=UTF-8
#
# Output the 50 most-used words from a text file, using NLTK FreqDist()
# (The text file must be in UTF-8 encoding.)
#
# Usage:
#
#   ./freqdist_top_words.py input.txt
#
# Sample output:
#
# et;8
# dolorem;5
# est;4
# aut;4
# sint;4
# dolor;4
# laborum;3
# ...
#
# Requires NLTK. Official installation docs: http://www.nltk.org/install.html
#
# I installed it on my Debian box like this:
#
# sudo apt-get install python-pip
# sudo pip install -U nltk
# python
# >>> import nltk
# >>> nltk.download('stopwords')
# >>> nltk.download('punkt')
# >>> exit()

import sys
import codecs
import nltk


vocab_file = "/opt3/home/lofi/word2vec_models/s2v.vocab"
input_file = "/opt3/home/pratima/thesis_final/models/s2v-final.txt"



fp = codecs.open(input_file, 'r', 'utf-8')

words = nltk.word_tokenize(fp.read())

# Remove single-character tokens (mostly punctuation)
words = [word for word in words if len(word) > 1]

# Remove numbers
words = [word for word in words if not word.isnumeric()]


# Calculate frequency distribution
fdist = nltk.FreqDist(words)

# Output top 50 words
with open(vocab_file, "w") as f:
    for word in fdist:
        if fdist[word]>4:
            f.write(u'{} {}\n'.format(word, fdist[word]))