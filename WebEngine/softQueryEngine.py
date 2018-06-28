import os
import glob
import math
import random
import re
import nltk
from bs4 import BeautifulSoup as bs

path = "./tokens"
outputPath = "./bm25ForSnippet"
pseudoRelOutputPath = "./pseudoRelBM25Ranking"
queryLocation = "./queries/cacm.query.txt"
commonWordsLoc = "./cacm_commonWords/common_words.dms"
noiseQueryLocation = "./noiseQueries/noiseQueries.txt"

withSoftMatch = "./noiseQueries/results/withSoftMatch"
withoutSoftMatch = "./noiseQueries/results/withoutSoftmatch"



tokenIndex = {}

docLength = {}

k1 = 1.2
k2 = 100
b = 0.75
TotalCourpusCount = 0
TotalWordCount = 0

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


def getAllQueries():
    allQueries = {}
    file = open(noiseQueryLocation, "r")

    count = 1
    for line in file:
        if "\n" in line:
            line = str.replace(line, "\n", "")
            allQueries[count] = line
            count += 1

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



def softMatchWord(word):

    if word in tokenIndex:
        return word

    startLetter = word[0]
    endLetter = word[len(word)-1]
    wordLen = len(word)

    allIndexWords = tokenIndex.keys()

    roughMatchItems = []
    for indexItem in allIndexWords:
        iFirst = indexItem[0]
        iLast = indexItem[len(indexItem)-1]

        if startLetter == iFirst and endLetter == iLast:
            # processing further
            if wordLen == len(indexItem):
                roughMatchItems.append(indexItem)

    probabilityDict = {}
    # further process for probability
    for w in roughMatchItems:
        freq = 0
        if w in tokenIndex:
            freqDocs = tokenIndex[w]
            for key in freqDocs:
                freq += freqDocs[key]
        probabilityDict[w] = freq

    res = probabilityDict
    for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
        return key

    return word

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



def noiseGenerator(query):

    # query = query.lower()
    queryWords  = query.split()

    queryLen = len(queryWords)

    alteredNum = 0

    queryWords.sort(key = len)

    count = 1
    for x in range(math.floor(queryLen * 0.4)):
        ran = random.randint(0, queryLen-1)
        word = queryWords[len(queryWords)-count]
        noisedWord = disturbWord(word)
        queryWords[len(queryWords)-count] = noisedWord
        alteredNum += 1
        count += 1

    # reforming the original query with the noised words
    noisedQry = ' '.join(queryWords)

    print("total altered words")
    print(alteredNum)

    return noisedQry


def disturbWord(word):

    wordLen = len(word)

    wordLst = list(word)
    if wordLen > 3:
        ran = random.randint(0, wordLen-1)
        if ran%2 == 0:
            tmp = wordLst[ran]
            if ran >= wordLen-2:
                wordLst[ran - 1] = tmp
            else:
                wordLst[ran + 1] = tmp
        else:
            tmp = wordLst[ran]
            if ran <= 1:
                wordLst[ran + 1] = tmp
            else:
                wordLst[ran - 1] = tmp

    elif wordLen == 3:
        wordLst[1] = wordLst[0]

    Nword = ''.join(wordLst)

    return Nword


def main():

    # for filename in os.listdir(os.getcwd()):
    count = 1
    global TotalCourpusCount
    global TotalWordCount

    # TotalCourpusCount = 0
    for filename in glob.glob(os.path.join(path, '*.txt')):
        processEachFile(filename)
        TotalCourpusCount += 1
        print(count)
        count += 1

    print("Index Creation done!\n\n")

    # process for total word count length count
    for key in docLength:
        TotalWordCount += docLength[key]



    allQueries = getAllQueries()


    # process for BM25 retrival method
    BM25Index = {}
    # query = "dark eclipse moon"
    avdl = getAvgDocLength()

    # # pre process query to bring in format
    # processedQueries = []
    # for itrKey in allQueries:
    #     processedQueries.append(preProcessEachQuery(allQueries[itrKey]))


    # softmatch the queries
    softMatchQ = {}
    for key in allQueries:
        q = allQueries[key]
        words = q.split()
        newWords = []
        for w in words:
            newWords.append(softMatchWord(w))
        newQ = " ".join(newWords)
        softMatchQ[key] = newQ

    # softMatchQ = allQueries



    # process BM25 results
    for key in softMatchQ:
        query = softMatchQ[key]
        BM25Index[query] = {}
        for file in files:
            BM25Index[query][file] = calculateBM25Fordoc(file, query, avdl)


    # write bm25 results in the folder
    # queryPos = 1
    for key in softMatchQ:
        query = softMatchQ[key]
        queryID  = key
        outputFile = open(withoutSoftMatch+"/"+str(key)+".txt", "w",encoding='utf-8')

        #writing query in the file first line
        # outputFile.write(query+"\n")
        res = BM25Index[query]
        rank = 1
        for key, value in sorted(res.items(), key=lambda item: (item[1], item[0]), reverse=True):
            if rank > 100 :
                break
            if not key:
                continue
            outputFile.write("%s\tQ0\t%s\t%s\t%s\tBM25System" % (queryID, key, rank, res[key]) + "\n")
            rank += 1

        # queryPos += 1


    return

main()




