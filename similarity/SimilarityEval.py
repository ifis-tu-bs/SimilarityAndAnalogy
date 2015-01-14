__author__ = 'Christoph'

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import brown, movie_reviews, treebank
from gensim.models import word2vec
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial.distance import *
from sklearn import svm

import os.path
import logging

# run this only if directly called and not when imported as a library

ic = None
w2v_model = None
clf = None
absPathToTestFiles = "/opt3/home/lofi/_pycharm/similarity/"

def loadGoldData(filename, correctorFactor):
    with open(filename, 'r') as file:
        lines = file.readlines()
        simGold = []
         # scan the string array
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            # skip all lines starting with blank, newline or 190
            if line.startswith("#"):
                continue
            if line.startswith("\n"):
                i += 1
                continue
            splits = line.split("\t")
            if len(splits) == 1:
                splits = line.split(" ")
            correctedsplits = []
            correctedsplits.append(splits[0])
            correctedsplits.append(splits[1])
            correctedsplits.append(float(splits[2]) * correctorFactor)
            simGold.append(correctedsplits)
        return simGold

def computeLinSimilarity(term1, term2):
    global ic
    if not ic:
        #ic = wordnet_ic.ic('ic-semcor.dat')
        ic = wordnet_ic.ic('ic-brown.dat')
    w1_syns = wn.synsets(term1)
    w2_syns = wn.synsets(term2)
    maxsim = 0
    for w1s in w1_syns:
        for w2s in w2_syns:
            try:
                sim = wn.lin_similarity(w1s, w2s, ic)
                if sim > maxsim:
                    maxsim = sim
            except Exception:
                pass
    return maxsim

def initW2V():
    global w2v_model
    model_filename="/opt3/.data-lofi/word2vec_google/GoogleNews-vectors-negative300.bin"
    #GoogleNews-vectors-negative300.bin.gz
    #freebase-vectors-skipgram1000.bin.gz
    #/opt3/home/lofi/word2vec_models/freebase-vectors-skipgram1000-en.bin.gz
    if (w2v_model is None):
        print("Loading W2V")
        # w2v_model= word2vec.Word2Vec(brown.sents())
        # w2v_model = word2vec.Word2Vec(movie_reviews.sents())
        w2v_model = word2vec.Word2Vec.load_word2vec_format(model_filename, binary=True)

def computeW2VSimilarity(term1, term2):
    global w2v_model
    initW2V()
    try:
        simvalue =  w2v_model.similarity(term1, term2)
    except KeyError:
        simvalue = 0
    if simvalue < 0:
        simvalue = 0
    return simvalue


def trainSVM():
    global clf
    gold_ws_r = loadGoldData(absPathToTestFiles+"EN-WS-353-related.txt", 1.0 / 10)
    gold_ws_s = loadGoldData(absPathToTestFiles+"EN-WS-353-similar.txt", 1.0 / 10)
    global w2v_model
    initW2V()
    vectorData = []
    labels = []
    for gold in gold_ws_s:
        if gold[2] > 0.5: # positive similarity tuple
            vectorData.append(w2v_model[gold[0]] - w2v_model[gold[1]])
            labels.append(1)
        if gold[2] < 0.5:
            vectorData.append(w2v_model[gold[0]] - w2v_model[gold[1]])
            labels.append(0)
  #  for gold in gold_ws_r:
 #       vectorData.append(w2v_model[gold[0]] - w2v_model[gold[1]])
  #      labels.append(0)
    clf = svm.SVC()
    clf.fit(vectorData, labels)





def runExperimentW2V(golddata, verbose):
    print("COMPUTING")
    similarity_vector = []
    reference_vector = []
    for g in golddata:
        #sim = computeLinSimilarity(g[0], g[1])
        sim = computeW2VSimilarity(g[0], g[1])
        similarity_vector.append(sim)
        reference_vector.append(g[2])
        if verbose:
            print("Similarity: %20s : %-20s measured: %5.3f correct:%5.3f " % (g[0], g[1], sim, g[2]))
            print("Class: %i " % clf.predict(w2v_model[g[0]]-w2v_model[g[1]])[0])
    correlation_p = pearsonr(reference_vector, similarity_vector)
    correlation_sp = spearmanr(reference_vector, similarity_vector)
    print("Correlation %4.3f  %4.3f " % (correlation_p[0], correlation_sp[0]))




if __name__ == '__main__':
    print("LOADING")
    gold_mc = loadGoldData(absPathToTestFiles+"EN-MC-30.txt", 1.0 / 4)
    gold_rg = loadGoldData(absPathToTestFiles+"EN-RG-65.txt", 1.0 / 4)
    gold_ws_r = loadGoldData(absPathToTestFiles+"EN-WS-353-related.txt", 1.0 / 10)
    gold_ws_s = loadGoldData(absPathToTestFiles+"EN-WS-353-similar.txt", 1.0 / 10)
    gold_ws = loadGoldData(absPathToTestFiles+"EN-WS-353-all.txt", 1.0 / 10)
    gold_men = loadGoldData(absPathToTestFiles+"MEN-full.txt", 1.0 / 50)
    trainSVM()
    print("mem")
    runExperimentW2V(gold_men, False)
    print("RG")
    runExperimentW2V(gold_rg, False)
    print("MC")
    runExperimentW2V(gold_mc, False)
    print("W2V")
    #runExperimentW2V(gold_ws, False)
    #runExperimentW2V(gold_ws_r, False)
    #runExperimentW2V(gold_ws_s, False)


