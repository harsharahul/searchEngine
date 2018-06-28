import os
import math
import operator
import re

stopWordList = []
queryDict = dict()

def process_score_file(query_id):
    queryDocumentResult = dict()
    QueryResult = open("BM25 Scores" + "/"+str(query_id)+".txt", "r")
    count = 0
    for resultLine in QueryResult.readlines():
        if count == 100:
            break
        else:
            resultLine = re.split(r'\t+', resultLine)
            queryId = resultLine[0]
            documentId = resultLine[2]
            if queryId in queryDocumentResult:
                queryDocumentResult[queryId].append(documentId)
            else:
                queryDocumentResult[queryId] = [documentId]
        count += 1
    return queryDocumentResult

def match(query, word):
    pattern_to_remove = re.compile(r'[,.!\"();:<>-_+=@#]')
    pattern_word = pattern_to_remove.sub('', word)
    query = query.split()

    for i in query:
        if i == pattern_word:
            return True

    return False

def returnSignificanceScore(sentence, query):
    sentence = sentence.split()
    min = 0
    max = 0
    count = 0
    result = 0.0
    for word in sentence:
        if match(query, word) and word not in stopWordList:
            min = sentence.index(word)
            break
    for i in range(len(sentence)-1, 0, -1):
        if match(query, sentence[i]) and sentence[i] not in stopWordList:
            max = i
            break
    for i in sentence[min: (max+1)]:
        if match(query, i) and i not in stopWordList:
            count+=1
    if(len(sentence[min:(max+1)])==0):
        result = 0
    else:
        result =  math.pow(count,2)/len(sentence[min:(max+1)])

    return result


def returnSnippet(queryId, queryDocumentResult):
    query = queryDict[int(queryId)]
    resultFileForQuery = queryDocumentResult[queryId]
    output_file = open("Snippets/"+queryId+".txt","w")
    output_file.write("Query:"+query)
    output_file.write("\n")
    for file in resultFileForQuery:
        file = open("Corpus/"+file+".txt", "r")
        sentences = file.read().split("\n")
        count = 1
        snippets = {}
        summary = []
        for sentence in sentences:
            snippets[sentence] = returnSignificanceScore(sentence, query)
        for i in sorted(snippets, key=snippets.get, reverse=True):
            if count < 5:
                summary.append(str(i))
                count += 1
        output_file.write("Document:"+file.name.split("/")[1].replace(".txt",""))
        output_file.write("\n")
        output_file.write("\n")
        for snip in summary:
            words = snip.split()
            if len(words) != 0:
                for word in words:
                    if match(query, word):
                        words[words.index(word)] = word.upper()
                        output_file.write(word.upper())
                        output_file.write(" ")
                    else:
                        output_file.write(word+" ")
                output_file.write("\n")
        output_file.write("*********************************************************************")
        output_file.write("\n")
    output_file.close()



def main():
    global queryDict, stopWordList
    stopListFile = open("common_words.txt", "r")
    for line in stopListFile.readlines():
        stopWordList.append(line.strip())
    stopListFile.close()

    queryFile = open("Queries.txt", "r")
    count = 1
    for query in queryFile.readlines():
        queryDict[count] = query
        count += 1

    for i in range(1, 65):
        query_id = str(i)
        processed_score_file =  process_score_file(query_id)
        returnSnippet(query_id, processed_score_file)

main()
