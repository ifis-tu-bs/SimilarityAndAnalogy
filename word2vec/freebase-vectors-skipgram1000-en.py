from gensim.models import word2vec
import os.path
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_filename="/opt3/home/lofi/word2vec_models/GoogleNews-vectors-negative300.bin.gz"
#GoogleNews-vectors-negative300.bin.gz
#freebase-vectors-skipgram1000.bin.gz
#/opt3/home/lofi/word2vec_models/freebase-vectors-skipgram1000-en.bin.gz

model = word2vec.Word2Vec.load_word2vec_format(model_filename, binary=True)

print ("model = word2vec.Word2Vec.load_word2vec_format('"+model_filename+"', binary=True)")
print ("Samples:")
print ("model.most_similar('/en/kenya', topn=10) # most similar to kenya")
print ("model.most_similar(['/en/adolf_hitler', '/en/russia'], ['/en/germany'], topn=10) # hitler - germany + russia")