__author__ = 'Athiq'
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial.distance import *

from Glove import eval,glove
import gensim


path = "/home/athiq/analogy"
def GloveSim(path, correctorFactor,verbose):
    similarity_vector = []
    reference_vector = []
    with open(path, 'r') as file:
        lines = file.readlines()
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
            print ("correctedsplits")
            correctedsplits.append(float(splits[2]) * correctorFactor)

            vocab = glove.build_vocab(correctedsplits)
            id2word = eval.make_id2word(vocab)
            cooccur = glove.build_cooccur(vocab, correctedsplits, window_size=10, min_count=None)
            W = glove.train_glove(vocab, cooccur, vector_size=5, iterations=5)
            W = eval.merge_main_context(W)
            word_id1 = vocab[correctedsplits[0]][0]
            word_id2 = vocab[correctedsplits[1]][0]
            vector1=W[word_id1]
            vector2=W[word_id2]
            cos= cosine(vector1,vector2)
            similarity_vector.append(cos)
            reference_vector.append(correctedsplits[2])
        correlation_p = pearsonr(reference_vector, similarity_vector)
        correlation_sp = spearmanr(reference_vector, similarity_vector)
        print(" %4.3f  %4.3f " % (correlation_p[0], correlation_sp[0]))

GloveSim(path,1.0/50,False)

