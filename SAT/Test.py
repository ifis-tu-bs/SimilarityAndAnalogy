#!/bin/env python3.0


from SAT.Challenge import Challenge
import random
from gensim.models import word2vec
import os.path
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def compute_similarity(option1, option2):
    try:
        sim = model.n_similarity([option1[0], option1[1]], [option2[0], option2[1]])
    except:
        sim = 0
    return sim

def loadSATchallanges():
    # read file into String array
    with open('SAT-package-V3.txt', 'r') as file:
        lines = file.readlines()

    challenges = []
    # scan the string array
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        # skip all lines starting with blank, newline or 190
        if line.startswith("#"):
            continue
        if line.startswith("\n"):
            i += 1
            continue

        # create a new challenge objects
        i -= 1
        challenge = Challenge()
        challenge.addOption(lines[i])
        challenge.addOption(lines[i+1])
        challenge.addOption(lines[i+2])
        challenge.addOption(lines[i+3])
        challenge.addOption(lines[i+4])
        challenge.addOption(lines[i+5])
        challenge.setCorrectIndex(lines[i+6])
        i += 7
        challenges.append(challenge)
       # print(challenge)

    del lines
    return challenges


# run this only if directly called and not when imported as a library 
if __name__ == '__main__':
    
    challenges=loadSATchallanges()

    # word2vec init
    model_filename="/opt3/home/lofi/word2vec_models/GoogleNews-vectors-negative300.bin.gz"
    #GoogleNews-vectors-negative300.bin.gz
    #freebase-vectors-skipgram1000.bin.gz
    #/opt3/home/lofi/word2vec_models/freebase-vectors-skipgram1000-en.bin.gz
    model = word2vec.Word2Vec.load_word2vec_format(model_filename, binary=True)
    #

    stats_correctchallenges = 0
    # test similarity for each option
    for challenge in challenges:
        bestoption_value = -1
        bestoption_index = 0
        for i in range(1,6):
           similarity=compute_similarity(challenge.options[0], challenge.options[i])
           if (similarity>bestoption_value):
                bestoption_value = similarity
                bestoption_index = i

        if (bestoption_index==challenge.correctIndex):
            stats_correctchallenges+=1
            print("SOLVED :", str(challenge))
        else:
            print("failed :", str(challenge))


    print("Ratio of correct challenges: ", stats_correctchallenges / float(len(challenges)))


