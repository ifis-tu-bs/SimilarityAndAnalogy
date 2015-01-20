from numpy.lib.function_base import diff

__author__ = 'Christoph'

from gensim.models import word2vec
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial.distance import *
import numpy

absPathToTestFiles = "/opt3/home/lofi/_pycharm/similarity/analogy/"
w2v = None
gold = None

def initW2V():
    global w2v
    model_filename="/opt3/.data-lofi/word2vec_google/GoogleNews-vectors-negative300.bin"
    #GoogleNews-vectors-negative300.bin.gz
    #freebase-vectors-skipgram1000.bin.gz
    #/opt3/home/lofi/word2vec_models/freebase-vectors-skipgram1000-en.bin.gz
    if (w2v is None):
        print("Loading W2V")
        # w2v_model= word2vec.Word2Vec(brown.sents())
        # w2v_model = word2vec.Word2Vec(movie_reviews.sents())
        w2v = word2vec.Word2Vec.load_word2vec_format(model_filename, binary=True)


## Loads a map of the question words txt. Map Identifier: Analogy Type Section, Map Entry: List of challanges, challange: 4 term list
def loadGoldData(filename):
    global gold
    with open(filename, 'r') as file:
        lines = file.readlines()
        gold = dict()
         # scan the string array
        i = 0
        currentIdentifier = ""
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            # skip all lines starting with blank, newline
            if line.startswith("#"):
                continue
            if line.startswith("\n"):
                i += 1
                continue
            # start a new identifier
            if line.startswith(": "):
                i += 1
                currentIdentifier = line.replace(": ", "")
                if not currentIdentifier in gold:
                    gold[currentIdentifier] = []
            splits = line.split(" ")
            gold[currentIdentifier].append(splits)


## computes diff vector statistics for a given section in the QW.txt file
def analyzeDiffVectors(sectionName):
    global gold, w2v
    analogies = gold[sectionName]
    # iterate over all analogies
    # and store analogons in a map with "a1_a2" as key and the term diff as value
    diffVectorsMap = {}
    diffVectors = []
    for analogy in analogies:
            if not analogy[0]+"_"+analogy[1] in diffVectorsMap:
                try:
                    t1 = w2v[analogy[0]]
                    t2 = w2v[analogy[1]]
                    diff = t1 - t2
                    diffVectorsMap[analogy[0]+"_"+analogy[1]] = diff
                    diffVectors.append(diff)
                except:
                    continue
            if not analogy[2]+"_"+analogy[3] in diffVectorsMap:
                try:
                    t1 = w2v[analogy[2]]
                    t2 = w2v[analogy[3]]
                    diff = t1 - t2
                    diffVectorsMap[analogy[2]+"_"+analogy[3]] = diff
                    diffVectors.append(diff)
                except:
                    continue
    mean_vec = numpy.mean(diffVectors, axis=0, dtype=numpy.float64)
    var_vec = numpy.var(diffVectors, axis=0, dtype=numpy.float64)
    print("%s : variance=%8.5f" % (sectionName, numpy.mean(var_vec)))
    return diffVectors


if __name__ == '__main__':
    global gold, w2v
    print("INIT")
    initW2V()
    print("LOADING")
    loadGoldData(absPathToTestFiles+"questions-words.txt")
    # diffVectors = analyzeDiffVectors("capital-common-countries")
    for section in gold.keys():
        diffVectors = analyzeDiffVectors(section)
    # done. analyze term diffs
    mean_vec = numpy.mean(diffVectors, axis=0, dtype=numpy.float64)
    var_vec = numpy.var(diffVectors, axis=0, dtype=numpy.float64)
    # syn0 is the data matrix of w2v
    # 3000000 x 300 for google news dataset
    ## this is the matrix projected to the average vector
    projection = numpy.abs(w2v.syn0.dot(mean_vec))
    ## this gets an index for a given word
    w2v.vocab["berlin"].index
