
import random
import math

import nltk
from bs4 import BeautifulSoup as bs
import re

saveLocation = "./noiseQueries/noiseQueries.txt"
queryLocation = "./queries/cacm.query.txt"

def getAllQueries():
    allQueries = {}
    file = open(queryLocation, "r")
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

def main():

    allQueries = getAllQueries()

    # process for BM25 retrival method
    BM25Index = {}
    # query = "dark eclipse moon"
    # avdl = getAvgDocLength()

    # pre process query to bring in format
    processedQueries = []
    for itrKey in allQueries:
        processedQueries.append(preProcessEachQuery(allQueries[itrKey]))

    # generate noise for the queries
    noiseQueries = []
    for query in processedQueries:
        noiseQueries.append(noiseGenerator(query))

    file = open(saveLocation, "w")

    for q in noiseQueries:
        file.write(q + "\n")

    file.close()

    return

def noiseGenerator(query):

    # query = query.lower()
    queryWords  = query.split()

    queryLen = len(queryWords)

    alteredNum = 0

    queryWords.sort(key = len)

    count = 1
    for x in range(math.floor(queryLen * 0.4)):
        ran = random.randint(0, queryLen-1)

        # iteration 1
        word = queryWords[len(queryWords)-count]
        noisedWord = disturbWord(word)
        queryWords[len(queryWords)-count] = noisedWord

        # iteration 2
        word = queryWords[len(queryWords) - count]
        noisedWord = disturbWord(word)
        queryWords[len(queryWords) - count] = noisedWord
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

main()




