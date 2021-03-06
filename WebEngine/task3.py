import os
import glob
import math
import re
import nltk
from bs4 import BeautifulSoup as bs

path = "./tokens"
outputPath = "./bm25Results"
pseudoRelOutputPath = "./pseudoRelBM25Ranking"
queryLocation = "./queries/cacm.query.txt"
commonWordsLoc = "./cacm_commonWords/common_words.dms"
stemmedQueryLocation = "./stemmed_corpus/cacm_stem.query.txt"

tokenIndex = {}
# biGramIndex = {}
# triGramIndex = {}

# unigramIndexPos = {}

docLength = {}

k1 = 1.2
k2 = 100
b = 0.75
TotalCourpusCount = 0

files = []

# calculated in TFID, used in Smoothed query likehood model
collectiontermfreq = {}


def getAvgDocLength():
    global sum
    sum = 0
    for key in docLength:
        sum += docLength[key]
    print("Total no. of words in collection is:",sum)
    return sum/len(docLength)


def getAllQueries(location):
    allQueries = {}
    file = open(location, "r")
    soup = bs(file.read(), "lxml")

    docSoup = soup.findAll("doc")

    for doc in docSoup:
        text = doc.text.split()
        queryNum = text[0]
        text[0] = ''
        query = ' '.join(text)
        print(text)
        allQueries[queryNum] = query

    return allQueries

def fetchAllStemQueries(location):

    allQueries = {}
    file = open(location, "r", encoding='utf-8')

    queryID = 1
    for line in file:
        if "\n" in file:
            line = str.replace(line,"\n","")
        allQueries[queryID] = line
        queryID += 1

    return allQueries

def preProcessEachQuery(query):

    regex = r"(?<!\d)[.,;:](?!\d)"

    pattern = r'''(?x)
                        (?:[A-Z]\.)+
                        | \w+(?:-\w)*
                        | \$?\d+(?:\.\d+)?%? '''

    allText = query.lower()

    allwords = allText.split()

    tokenLst = []
    for word in allwords:
        # checking if this is only digits then do so ie; no text
        if (re.search('[a-zA-Z]', word) == None):
            result = re.sub(regex, "", word, 0)
            for char in ['{', '}', '(', ')', '[', ']']:
                result = str.replace(result, char, "")

            if result.startswith(","):
                result = result[1:]
            if result.endswith(","):
                result = result[:-1]
            tokenLst.append(result)
        # this condition is for hyphenated words
        elif "-" in word:
            tokenLst.append(word)
        # All other words , do this
        else:
            tokenize_words = nltk.regexp_tokenize(word, pattern)
            tokenLst += tokenize_words

    return ' '.join(tokenLst)





def processEachFile(filename):
    global files
    global tokenIndex
    global docLength
    file = open(filename,"r",encoding='utf-8')

    tokens = []

    fullText = ""
    for line in file:
        if '\n' in line:
            line = line[:-1]
        tokens.append(line)
        fullText = fullText + " " + line

    fileparts = filename.split('/')

    filename = fileparts[len(fileparts) -1]
    subparts = filename.split('.')

    filename = subparts[0]


    # Saving the document length -> to br used for BM25 retrival
    docLength[filename] = len(tokens)
    files.append(filename)
    #

    formIndex(filename, tokenIndex, tokens)


    return

def formNgramTokens(n, text):
    from nltk import ngrams
    ngrams = ngrams(text.split(), n)
    ngramTokens = []

    for grams in ngrams:
        if n == 2:
            ngramTokens.append(grams[0]+" "+grams[1])
        if n == 3:
            ngramTokens.append(grams[0]+" "+grams[1]+ " " +grams[2])


    return ngramTokens

def formIndex(filename, index, tokens):

    for token in tokens:
        if '\n' in token:
            token = token[:-1]

        if token in index:
            if filename in index[token]:
                index[token][filename] += 1
            else:
                index[token].update({filename: 1})
        else:
            # docID[filename] = 1
            index[token] = {filename: 1}




def calculateBM25Fordoc(file, query, avdl):
    queryTokens = str.split(query)
    queryKeys = {}
    N = len(docLength)
    # getting qf for each query token
    for eachToken in queryTokens:
        if eachToken in queryKeys:
            queryKeys[eachToken] += 1
        else:
            queryKeys[eachToken] = 1

    # calculating BM25 for each query term
    bm25 = 0
    K = getK(file, docLength, k1, b, avdl)
    for eachQueryTerm in queryTokens:
        qfi = queryKeys[eachQueryTerm]

        if eachQueryTerm in tokenIndex:
            ni = len(tokenIndex[eachQueryTerm])
        else:
            ni = 0

        if eachQueryTerm in tokenIndex:
            if file in tokenIndex[eachQueryTerm]:
                fi = tokenIndex[eachQueryTerm][file]
            else:
                fi = 0
        else:
            fi = 0

        R = 0
        ri = 0
        bm25 += getBM25(eachQueryTerm, file, qfi, K, k1, k2, N, ni, fi, R, ri)

    return bm25


def getK(file, docLength, k1, b, avdl):
    return k1 * ((1-b) + b * (docLength[file]/avdl))

def getBM25(eachQueryTerm, file, qfi, K, k1, k2, N, ni, fi, R, ri):
    return math.log(((ri + 0.5) / (R-ri+0.5)) / ((ni-ri+0.5) / (N-ni-R+ri+0.5))) * (((k1+1)*fi)/(K+fi)) * (((k2+1)*qfi)/(k2+qfi))





def loadCommonWords():
    commonWords = []
    file = open(commonWordsLoc, "r")
    for line in file:
        if "\n" in line:
            line = str.replace(line, "\n",'')
            if line.isspace():
                continue
        commonWords.append(line)
    return commonWords

def getTokens(filename):
    file = open(filename, "r")
    tokens = []
    for line in file:
        if "\n" in line:
            line = str.replace(line,"\n","")
        tokens.append(line)

    return tokens

def pseudoRelevanceFeedback(query, ranks):

    newQuery = query.split()
    for x in range(0,5):
        filename = ranks[x][0]
        rank = ranks[x][1]


        commonWordTokens = getTokens("./tokens/"+filename+".txt")

        frequecyDict = {}
        for line in commonWordTokens:
            if line in frequecyDict:
                frequecyDict[line] += 1
            else:
                frequecyDict[line] = 1

        commonWords = loadCommonWords()

        for word in commonWords:
            if word in frequecyDict:
                del frequecyDict[word]

        count = 0
        for key, value in sorted(frequecyDict.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if count > 5:
                break
            newQuery.append(key)
            count += 1

    return " ".join(newQuery)

def newTfIdf(file, query):
    queryWords = query.split()
    N = len(docLength)
    sumTfIdf = 0
    for word in queryWords:
        if word in tokenIndex:
            if file in tokenIndex[word]:
                tf = tokenIndex[word][file] / docLength[file]

                df = len(tokenIndex[word])

                idf = math.log(N / df)

                sumTfIdf += tf * idf

    return sumTfIdf


def newSmoothQ(file, query, C):
    lamda = 0.35
    queryWords = query.split()

    smoothScore = 0
    for word in queryWords:

        cqi = 0
        if word in tokenIndex:
            invertedIndexForTerm = tokenIndex[word]
            for key in invertedIndexForTerm:
                cqi += invertedIndexForTerm[key]

            if word in tokenIndex:
                if file in tokenIndex[word]:
                    fqid = tokenIndex[word][file]
                    dLen = docLength[file]
                else:
                    fqid = 0
                    dLen = docLength[file]
            else:
                fqid = 0
                dLen = docLength[file]

            a = (1-lamda) * (fqid/dLen)
            b = lamda * (cqi/C)

            smoothScore += math.log(a+b)
        else:
            # this case when the term is not there in the entire  corpus
            smoothScore += 0

    return smoothScore

def processStemedCorpus():
    file = open("./stemmed_corpus/cacm_stem.txt", "r")

    allText = file.read()
    allDocs = allText.split("#")

    for doc in allDocs:
        if doc.isspace():
            continue
        if not doc:
            continue


        # removing first empty space
        doc = doc[1:]

        parts = doc.split()
        # fetching the doc id
        docID = int(parts[0])


        # delete the docID
        del parts[0]
        doc = " ".join(parts)
        # # removing the doc ID from the corpus
        # doc = doc[1:]

        # processing
        indx = doc.rfind("pm")
        if indx >= 0:
            doc = doc[: (indx + 2)]
        else:
            indx = doc.rfind("am")
            doc = doc[: (indx + 2)]



        outputfile = open("./stemmed_corpus/tokens/CACM-%04d.txt" % docID, "w")

        words = doc.split()

        for word in words:
            if "\n" in word:
                word = str.replace(word,"\n","")
            if word.isspace():
                continue
            if not word:
                continue
            outputfile.write(word)
            outputfile.write("\n")

        outputfile.flush()
        outputfile.close()


    return

def main():

    stopping = False
    stemming = True

    # for filename in os.listdir(os.getcwd()):
    count = 1
    global TotalCourpusCount

    # TotalCourpusCount = 0
    if stemming:
        processStemedCorpus()
        for filename in glob.glob(os.path.join("./stemmed_corpus/tokens", '*.txt')):
            processEachFile(filename)
            TotalCourpusCount += 1
            print(count)
            count += 1
    else:
        for filename in glob.glob(os.path.join(path, '*.txt')):
            processEachFile(filename)
            TotalCourpusCount += 1
            print(count)
            count += 1

    print("Index Creation done!\n\n")

    # removing stopwords from Index
    if stopping:
        commonWords = loadCommonWords()

        for word in commonWords:
            if word in tokenIndex:
                del tokenIndex[word]




    # file1 = open("unigramIndex.txt","w",encoding='utf-8')
    #
    # for key in tokenIndex:
    #     file1.write(key+" -> " + str(tokenIndex[key]))
    #     file1.write("\n")

    #print(tokenIndex)

    if stemming:
        allQueries = fetchAllStemQueries(stemmedQueryLocation)
    else:
        allQueries = getAllQueries(queryLocation)


    # process for BM25 retrival method
    BM25Index = {}
    # query = "dark eclipse moon"
    avdl = getAvgDocLength()

    processedQueries = []
    for itrKey in allQueries:
        processedQueries.append(preProcessEachQuery(allQueries[itrKey]))

    for query in processedQueries:
        BM25Index[query] = {}
        for file in files:
            BM25Index[query][file] = calculateBM25Fordoc(file, query, avdl)




    # write bm25 results in the folder
    queryPos = 1
    for query in processedQueries:
        outputFile = open("./task3/bm25/"+str(queryPos)+".txt", "w",encoding='utf-8')

        #writing query in the file first line
        outputFile.write(query+"\n")
        res = BM25Index[query]
        rank = 1
        for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if rank > 100:
                break
            if not key:
                continue
            outputFile.write("%s Q0 %s %s %s BM25System" % (queryPos, key, rank, res[key]) + "\n")
            rank += 1

        queryPos += 1

    # # processing rank for pseudo revelance feedback
    # pseudoProcessedBM25Index = {}
    # pseudoExpandedQueryDict = {}
    # for query in processedQueries:
    #     results = BM25Index[query]
    #     rankArr = []
    #     for key, value in sorted(results.items(), key=lambda item: (item[1], item[0]), reverse=True):
    #         if not key:
    #             continue
    #         rankArr.append((key, results[key]))
    #
    #     pseudoProcessedBM25Index[query] = {}
    #     pseudoExpandedQueryDict[query] = pseudoRelevanceFeedback(query, rankArr)
    #     for file in files:
    #         pseudoProcessedBM25Index[query][file] = calculateBM25Fordoc(file, pseudoExpandedQueryDict[query], avdl)
    #
    # # write pseudorelevance results in the folder
    # queryPos = 1
    # for query in processedQueries:
    #     outputFile = open(pseudoRelOutputPath + "/" + str(queryPos) + ".txt", "w", encoding='utf-8')
    #
    #     # writing query in the file first line
    #     outputFile.write("Original query:\n")
    #     outputFile.write(query + "\n")
    #     outputFile.write("Expanded query:\n")
    #     outputFile.write(pseudoExpandedQueryDict[query] + "\n")
    #
    #     res = pseudoProcessedBM25Index[query]
    #     rank = 1
    #     for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
    #         if not key:
    #             continue
    #         outputFile.write("%s Q0 %s %s %s PseudoRel_BM25System" % (queryPos, key, rank, res[key]) + "\n")
    #         rank += 1
    #
    #     queryPos += 1


    # process for tfIdf
    TFIDFIndex = {}
    for query in processedQueries:
        TFIDFIndex[query] = {}
        for file in files:
            TFIDFIndex[query][file] = newTfIdf(file, query)

    # write tfid results in the folder
    queryPos = 1
    for query in processedQueries:
        outputFile = open("./task3/tfidf/"+str(queryPos)+".txt", "w",encoding='utf-8')

        #writing query in the file first line
        outputFile.write(query+"\n")
        res = TFIDFIndex[query]
        rank = 1
        for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if rank > 100:
                break
            if not key:
                continue
            outputFile.write("%s Q0 %s %s %s TFIDFSystem" % (queryPos, key, rank, res[key]) + "\n")
            rank += 1

        queryPos += 1

    # process for smoothed query expansion
    smoothIndex = {}
    CorpusCount = 0
    for key in docLength:
        CorpusCount += docLength[key]


    for query in processedQueries:
        smoothIndex[query] = {}
        for file in files:
            smoothIndex[query][file] = newSmoothQ(file, query, CorpusCount)

    # write smooth results in the folder
    queryPos = 1
    for query in processedQueries:
        outputFile = open("./task3/smoothQ/" + str(queryPos) + ".txt", "w", encoding='utf-8')

        # writing query in the file first line
        outputFile.write(query + "\n")
        res = smoothIndex[query]
        rank = 1
        for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if rank>100:
                break
            if not key:
                continue
            outputFile.write("%s Q0 %s %s %s SMOOTHQSystem" % (queryPos, key, rank, res[key]) + "\n")
            rank += 1

        queryPos += 1

    return

main()






