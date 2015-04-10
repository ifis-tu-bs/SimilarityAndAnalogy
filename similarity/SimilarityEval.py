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
w2v = None
clf = None
## Path for remote execution
## When executing remotly, also do a proper mapping between local files and remote files in the config
## Path for remote execution Christoph
absPathToTestFiles = "/home/lofi/_pycharm_projects/similarity/"
## path for local execution Christoph
# absPathToTestFiles = "C:/Users/Christoph/SkyDrive/Documents/_pycharm_projects/analogy/similarity/"

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
                # sim = wn.lin_similarity(w1s, w2s, ic)
                sim = wn.jcn_similarity(w1s, w2s, ic)
                if sim > maxsim:
                    maxsim = sim
            except Exception:
                pass
    return maxsim

def initW2V():
    global w2v
    model_filename="/home/lofi/word2vec_models/GoogleNews-vectors-negative300.bin.gz"
    #GoogleNews-vectors-negative300.bin.gz
    #"/opt3/.data-lofi/word2vec_google/GoogleNews-vectors-negative300.bin"
    #freebase-vectors-skipgram1000.bin.gz
    #/opt3/home/lofi/word2vec_models/freebase-vectors-skipgram1000-en.bin.gz
    if (w2v is None):
        print("Loading W2V")
        # w2v= word2vec.Word2Vec(brown.sents())
        # w2v = word2vec.Word2Vec(movie_reviews.sents())
        w2v = word2vec.Word2Vec.load_word2vec_format(model_filename, binary=True)

def computeW2VSimilarity(term1, term2):
    global w2v
    initW2V()
    try:
        simvalue =  w2v.similarity(term1, term2)
    except KeyError:
        simvalue = 0
    if simvalue < 0:
        simvalue = 0
    return simvalue


def trainSVM():
    global clf
    gold_ws_r = loadGoldData(absPathToTestFiles+"EN-WS-353-related.txt", 1.0 / 10)
    gold_ws_s = loadGoldData(absPathToTestFiles+"EN-WS-353-similar.txt", 1.0 / 10)
    global w2v
    initW2V()
    vectorData = []
    labels = []
    for gold in gold_ws_s:
        if gold[2] > 0.5: # positive similarity tuple
            vectorData.append(w2v[gold[0]] - w2v[gold[1]])
            labels.append(1)
        if gold[2] < 0.5:
            vectorData.append(w2v[gold[0]] - w2v[gold[1]])
            labels.append(0)
  #  for gold in gold_ws_r:
 #       vectorData.append(w2v_model[gold[0]] - w2v_model[gold[1]])
  #      labels.append(0)
    clf = svm.SVC()
    clf.fit(vectorData, labels)





def runExperimentW2V(datasetlabel, golddata, verbose):
    similarity_vector = []
    reference_vector = []
    for g in golddata:
        sim = computeLinSimilarity(g[0], g[1])
        #sim = computeW2VSimilarity(g[0], g[1])
        similarity_vector.append(sim)
        reference_vector.append(g[2])
        if verbose:
            print("Similarity: %20s : %-20s measured: %5.3f correct:%5.3f " % (g[0], g[1], sim, g[2]))
            print("Class: %i " % clf.predict(w2v[g[0]]-w2v[g[1]])[0])
    correlation_p = pearsonr(reference_vector, similarity_vector)
    correlation_sp = spearmanr(reference_vector, similarity_vector)
    print("%s : %4.3f  %4.3f " % (datasetlabel, correlation_p[0], correlation_sp[0]))




if __name__ == '__main__':
    print("LOADING")
    gold_mc = loadGoldData(absPathToTestFiles+"EN-MC-30.txt", 1.0 / 4)
    gold_rg = loadGoldData(absPathToTestFiles+"EN-RG-65.txt", 1.0 / 4)
    gold_ws_r = loadGoldData(absPathToTestFiles+"EN-WS-353-related.txt", 1.0 / 10)
    gold_ws_s = loadGoldData(absPathToTestFiles+"EN-WS-353-similar.txt", 1.0 / 10)
    gold_ws = loadGoldData(absPathToTestFiles+"EN-WS-353-all.txt", 1.0 / 10)
    gold_men = loadGoldData(absPathToTestFiles+"MEN-full.txt", 1.0 / 50)
    gold_simlex = loadGoldData(absPathToTestFiles+"SimLex-999.txt", 1.0 / 50)
    # trainSVM()
    runExperimentW2V("MC30", gold_mc, False)
    runExperimentW2V("RG65", gold_rg, False)
    runExperimentW2V("MEM", gold_men, False)
    runExperimentW2V("W353",gold_ws, False)
    runExperimentW2V("W353-s", gold_ws_s, False)
    runExperimentW2V("W353-r", gold_ws_r, False)
    runExperimentW2V("simlex", gold_simlex, False)


