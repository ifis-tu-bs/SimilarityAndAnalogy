__author__ = 'Christoph'

absPathToTestFiles = "/opt3/home/lofi/_pycharm/similarity/analogy/"
w2v_model = None

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


## Loads a map of the question words txt. Map Identifier: Analogy Type Section, Map Entry: List of challanges, challange: 4 term list
def loadGoldData(filename):
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
        return gold


if __name__ == '__main__':
    print("INIT")
    initW2V()
    print("LOADING")
    gold = loadGoldData(absPathToTestFiles+"questions-words.txt")

