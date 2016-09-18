import w2vAida
import analogy_completion


#model_file = "D:/data/analogy/pratima_w2v_models/aida.model.bin"
model_file = "D:/data/analogy/pratima_w2v_models/s2v.model.bin"
golddata_file = "../testData/AGS/AGS-V02.txt"
output_file = "./result/test.txt"

def loadGoldData(dataset):
    with open(dataset, 'r') as file:
        lines = file.readlines()
        simGold = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if line.startswith(":"):
                continue
            if line.startswith("\n"):
                i += 1
                continue
            splits = line.split(" ")
            correctedsplits = [ ]
            correctedsplits.append(splits[0])
            correctedsplits.append(splits[1])
            correctedsplits.append(splits[2])
            correctedsplits.append(splits[3])
            correctedsplits.append(splits[4])
            correctedsplits.append(splits[5])
            correctedsplits.append(splits[6])
            correctedsplits.append(splits[7])
            #correctedsplits.append(splits[8])
            simGold.append(correctedsplits)
            #print(correctedsplits)
        return simGold

analogydataset = loadGoldData(golddata_file)
model=w2vAida.Word2Vec.load_word2vec_format(model_file, binary=True)

duplicatecount = 0.0
count = 0.0
countlines = 0.0
previouscheckvalue = 0.0
total = 0.0
nextval = 0

for value in analogydataset:
    # if(value[6]==word):
    r_similarity = model.n_similarity_new([value[0], value[1]], [value[2], value[3]])
    print("{}:{} :: {}:{}  >> {}".format(value[0], value[1], value[2], value[3], r_similarity))

    similarityrating = value[5]
    countlines += 1
    originalRating = value[4]
    originalcheckvalue = value[0]
    if (originalRating >= '4.5' and similarityrating >= '0.70'):
        count += 1
        if (previouscheckvalue == originalcheckvalue):
            count = duplicatecount
            duplicatecount = count
            previouscheckvalue = originalcheckvalue
        # print(originalcheckvalue)
        if nextval != originalcheckvalue:
            # print(nextval,originalcheckvalue)
            total += 1
        nextval = originalcheckvalue