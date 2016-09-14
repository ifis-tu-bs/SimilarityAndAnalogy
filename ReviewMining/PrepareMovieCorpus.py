from Cython.Plex.Regexps import RE
from blaze.utils import get

__author__ = 'Christoph'
import gzip
import os.path
from gensim import corpora, models
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



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

dirname = "AMAZON REVIEWS/"
dirname_processed = "AMAZON REVIEWS/processed/"
filename = "movies.txt.gz"
dictionary = None
corpus = None


class ReviewCorpus(object):
  def __iter__(self):
    global filename, dirname
   #  dictionary = corpora.Dictionary()
    for review in parse(dirname + filename):
      yield dictionary.doc2bow(getTokensFromEntry(review))



## Parse creates an iterateable generator containing string maps from a review file. Map keys see above.
def parse(absFilename):
  f = gzip.open(absFilename, 'rb')
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
    text = entry.get("review/text")
    if text is None:
        return []
    tokens=text.lower().split()
    return tokens

## copies all reviews of a single product into one review text (so, you get one huge review instead of multiple smaller ones).
## Meta data besides product id and title is of course lost
def groupReviewsByProductAndStore():
    f = gzip.open(dirname_processed+".grouped.gz", 'wb')
    lastProductId = " "
    lastDocument = None
    for review in parse(dirname + filename):
        # oh, we encounter a new review!
        if not lastProductId == review.get("product/productId"):
            if lastDocument is not None:
                f.write(bytes("product/productId : "+lastProductId+"\n","UTF-8"))
                f.write(bytes("product/title : "+lastProductTitle+"\n","UTF-8"))
                f.write(bytes("review/text: "+lastDocument+"\n\n","UTF-8"))
            lastDocument = review.get("review/text")
            lastProductId = review.get("product/productId")
            if lastProductId is None:
                lastProductId = "unknownID"
            lastProductTitle = review.get("product/title")
            if lastProductTitle is None:
                lastProductTitle = "unknownTitle"
        else:
            lastDocument += " "+review.get("review/text")
    f.close()

def extractIDs():
    # f = gzip.open(dirname_processed+".idonly.gz", 'wb')
    f = open(dirname_processed+filename+".idonly.txt", 'wb', 50)
    lastProductId = None
    lastDocument = None
    lastProductTitle = None
    for review in parse(dirname + filename):
        # oh, we encounter a new review!
        if not lastProductId == review.get("product/productId"):
            if not(lastProductTitle is None or lastProductTitle is "unknownTitle"):
                f.write(bytes(lastProductId+"\t"+lastProductTitle+"\n"))
                print(lastProductId+"\t"+lastProductTitle)
            else:
                print("Unknown title")
            lastProductId = review.get("product/productId")
            if lastProductId is None:
                lastProductId = "unknownID"
            lastProductTitle = review.get("product/title")
            if lastProductTitle is None:
                lastProductTitle = "unknownTitle"
        else:
            pass
    f.close()


if __name__ == '__main__':
    #for entry in reviews:
    #    print(entry.get("product/title"))
    # groupReviewsByProductAndStore()
    print("Running...")
    extractIDs()

    #for entry in reviews:
    #    print(entry.get("product/title"))