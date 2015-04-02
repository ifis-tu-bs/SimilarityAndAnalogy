from Cython.Plex.Regexps import RE
from blaze.utils import get

__author__ = 'Christoph'
import gzip
import os.path
from gensim import corpora, models, utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

filename = "./Xuan Review/amazoncorpus"

##
# FILES FROM http://snap.stanford.edu/data/web-Amazon-links.html
# product/productId: asin, e.g. amazon.com/dp/B00006HAXW
# product/title: title of the product
# product/price: price of the product
# review/userId: id of the user, e.g. A1RSDE90N6RSZF
# review/profileName: name of the user
# review/helpfulness: fraction of users who found the review helpful
# review/score: rating of the product
# review/time: time of the review (unix time)
# review/summary: review summary
# review/text: the text
##


filename = "AMAZON REVIEWS/Video_Games.txt.gz.grouped.gz"
dictionary = None
corpus = None
stopwords = None


class ReviewCorpus(object):
  def __iter__(self):
    global filename
   #  dictionary = corpora.Dictionary()
    for review in parse(filename):
      tokens= getTokensFromEntry(review)
      yield dictionary.doc2bow(tokens)



## Parse creates an iterateable generator containing string maps from a review file. Map keys see above.
def parse(filename):
  f = gzip.open(filename, 'rt', encoding='utf-8')
  entry = {}
  entry["review/text"] = " "
  counter = 0
  # iterate over all json objects, and create new maps with values
  for l in f:
    l = str(l.strip())
    colonPos = l.find(':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    entryName = str(l[:colonPos])
    entryValue = str(l[colonPos+2:])
    entry[entryName] = entryValue
  yield entry


## returns the content tokens from an entry map (i.e., the review text)
def getTokensFromEntry(entry):
    if stopwords is None:
        pass
    text = entry.get("review/text")
    if text is None:
        print("Empty Document")
        return ["None"]
    tokens= utils.tokenize(text, lower=True, errors='ignore')
    return tokens


# loads dictionary from disk or creates & saves a new one
def loadDictionary():
    global filename, dictionary
    if os.path.isfile(filename+".dictionary"):
        dictionary = corpora.Dictionary.load(filename+".dictionary")
    else:
        reviews = parse(filename)
        dictionary = corpora.Dictionary()
        # scan through all reviews, collect all text belonging to the same reviews in a single document
        for review in reviews:
            documents=[]
            documents.append(getTokensFromEntry(review))
            dictionary.add_documents(documents)
        dictionary.save(filename+".dictionary")

# loads a corpus from disk or creates a new one
def loadCorpus():
    global filename, dictionary, corpus
    if os.path.isfile(filename+".mm"):
        corpus = corpora.MmCorpus.load(filename+".mm")
    else:
        corpusRaw = ReviewCorpus()
        #model = models.TfidfModel(corpusRaw)
        #model = models.ldamulticore.LdaMulticore(corpus=corpusRaw, id2word=dictionary, num_topics=100, workers=2)
        model = models.ldamulticore.LdaModel(corpus=corpusRaw, id2word=dictionary, num_topics=100)
        corpus = model[corpus]
        corpora.MmCorpus.serialize(filename+".mm", corpus)



if __name__ == '__main__':
    #for entry in reviews:
    #    print(entry.get("product/title"))
    loadDictionary()
    print(dictionary)
    loadCorpus()
    print(corpus)

    #for entry in reviews:
    #    print(entry.get("product/title"))