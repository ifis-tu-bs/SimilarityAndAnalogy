

__author__ = 'Christoph'

import numpy
import gzip
import os.path
from gensim import corpora, models, utils, matutils
from nltk.corpus import stopwords
from os import listdir
import codecs
import string
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.feature_extraction.text import TfidfVectorizer

dictionary = None
cachefilename = "./Xuan Review/ReviewCache"
fStopwords = False



class ReviewCorpus(object):
    def __iter__(self):
        #  dictionary = corpora.Dictionary()
        for review in readAllFiles():
            tokens = tokenize(review)
            yield dictionary.doc2bow(tokens)

    def __len__():
        return len(listdir("./Xuan Review/review"))

# default function for tokenization
def tokenize(text):
    return utils.simple_preprocess(text, deacc=True, min_len=2, max_len=15)

# strips stopwords and punctuation
def readFile(filename, remove_stopwords=True, lower=True):
    text = u""
    sw = stopwords.words('english')
    punctDict = {ord(c): None for c in string.punctuation}
    with codecs.open(filename,'r','UTF-8') as inFile:
        for line in inFile:
            line=line.strip()
            # break first line of review
            if "        By        " in line:
                line = line.split("        By        ")[0]
                if "people found the following review helpful" in line:
                    line = line.split("people found the following review helpful")[1]
            # remove comment counter
            if line.endswith("Comment") or line.endswith("Comments"):
                continue
            # convert to lower case
            if lower:
                line = line.lower()
            # remove all punctuation, kill words shorter than 3 characters and stopwords.
            newline=" ".join([word for word in line.translate(punctDict).split() if not remove_stopwords or (word not in sw)])
            text=text+newline+"\n"
    return text


def readAllFiles():
    for f in listdir("./Xuan Review/review"):
        if f.endswith(".txt"):
            file = readFile("./Xuan Review/review/"+f, remove_stopwords=fStopwords, lower=True)
            yield file



def doLDA_Sklearn():
    global cachefilename
    number_of_topics = 60
    algorithm = "LSI"
    ###
    print("init")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    mtx_tfidf=vectorizer.fit_transform(readAllFiles())
    print("tfidf - done."+str(mtx_tfidf.shape)+" documents" )
    print("transform to gensim ")
    corpus_tfidf = matutils.Sparse2Corpus(mtx_tfidf, documents_columns=False)
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
    ### LSI
    if algorithm=="LSI":
        if os.path.isfile(cachefilename+"_"+str(number_of_topics)+"_LSI"):
            topicmodel = models.LsiModel.load(cachefilename+"_"+str(number_of_topics)+"_LSI")
        else:
            topicmodel = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=number_of_topics)
            topicmodel.save(cachefilename+"_"+str(number_of_topics)+"_LSI")
        corpus_tm=topicmodel[corpus_tfidf]
        dense_corpus=matutils.corpus2dense(corpus_tm, num_terms=number_of_topics).transpose()
        print("Finished dense corpus: "+str(dense_corpus.shape))
        numpy.savetxt(cachefilename+"_"+str(number_of_topics)+"_LSI.CSV", dense_corpus, delimiter=",")
    #lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=20)
    #lsi.print_topics(20)
    #corpus_lsi = lsi[corpus_tfidf]
    ### LDA
    if algorithm=="LDA":
        if os.path.isfile(cachefilename+"_"+str(number_of_topics)+"_LDA"):
            lda = models.ldamulticore.LdaModel.load(cachefilename+"_"+str(number_of_topics)+"_LDA")
        else:
            lda = models.ldamulticore.LdaModel(corpus=corpus_tfidf, id2word=id2word, num_topics=number_of_topics,  passes = 10)
            lda.save(cachefilename+"_"+str(number_of_topics)+"_LDA")
        print(lda.show_topics(num_topics=number_of_topics, num_words=10, formatted=True))
        corpus_lda=lda[corpus_tfidf]
        dense_corpus=matutils.corpus2dense(corpus_lda, num_terms=number_of_topics).transpose()
        print("Finished dense corpus: "+str(dense_corpus.shape))
        numpy.savetxt(cachefilename+"_"+str(number_of_topics)+"_LDA.CSV", dense_corpus, delimiter=",")



if __name__ == '__main__':
    doLDA_Sklearn()
