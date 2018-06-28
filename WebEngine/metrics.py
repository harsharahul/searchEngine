from bs4 import BeautifulSoup as bs
import os
import glob

relFileLocation = "./cacm_relation/cacm.rel.txt"
queriesLocation = "./queries/cacm.query.txt"
rankLocation = "./metrics/input"
outputLocation = "./metrics/output"

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


def loadRelData():
    file = open(relFileLocation, "r")

    relDict = {}
    for line in file:
        parts = line.split()
        queryId = parts[0]
        docId = parts[2]
        if queryId in relDict:
            relDict[queryId].append(docId)
        else:
            relDict[queryId] = [docId]

    file.close()

    return relDict


def getRanks(location, rankDict):
    file = open(location,"r")

    fileParts = location.split("/")
    subpart = fileParts[len(fileParts)-1].split(".")
    queryKey = subpart[0]

    # rankDict = {}
    ranks = []
    for line in file:
        parts = line.split()
        if len(parts) == 6:
            if parts[0].isdigit() and not parts[1].isdigit():
                ranks.append(parts)
        # else:
        #     rankDict[line] = ''

    # for key in rankDict:
    #     rankDict[key] = ranks

    rankDict[queryKey] = ranks

    return rankDict


def main():

    relData = loadRelData()
    # get all queries
    allQueries = getAllQueries(queriesLocation)

    rankDict = {}
    for filename in glob.glob(os.path.join(rankLocation, '*.txt')):
        getRanks(filename, rankDict)


    # print(ranks)
    results = {}
    for key in rankDict:
        eachRankset = rankDict[key]
        results[key] = getPrecisionRecall(key, eachRankset, relData, allQueries)


    # writing statistics

    file = open(outputLocation+"/Stats.txt","w")
    file.write("Statistics for Analysed system are :\n\n")
    MAPsum = 0
    MRRsum = 0

    count = 0
    for key in results:
        if not results[key]:
            continue
        file.write("Query ID -> " + key + "\n")
        if len(results[key][0]) < 1:
            continue
        MAPsum += results[key][0]["AP"]
        file.write("Average Precision is : " + str(results[key][0]["AP"])+"\n")

        if "RR" in results[key][0]:
            MRRsum += results[key][0]["RR"]
            file.write("Recriprocal rank is : " + str(results[key][0]["RR"]) + "\n")
        else:
            MRRsum += 0


        count += 1

    MAP = MAPsum/count
    MRR = MRRsum/count


    file.write("\n\nMean Average precision for the system is : " + str(MAP))
    file.write("\n\n")
    file.write("Mean Reciprocal Rank for the system is : " + str(MRR))


    return

def getPrecisionRecall(qryID, rankset, relData, allQueries):

    if not qryID in relData:
        return

    file = open(outputLocation+"/"+qryID+".txt", "w")

    file.write("Rank\t\tRelevance\t\tPrecision\t\tRecall\n")

    relcount = len(relData[qryID])
    precision = {}
    recall = {}
    precisionCounter = 0
    recallCounter = 0

    avgPrecisionArr = []
    for eachRank in rankset:
        # format of result
        # 1 Q0 CACM-1161 2 23.034889652990653 BM25System
        rank = int(eachRank[3])
        docID = eachRank[2]
        if qryID in relData:
            if docID in relData[qryID]:
                precisionCounter += 1
                recallCounter +=1

                #  Getting RR
                if precisionCounter == 1:
                    precision["RR"] = 1/rank

                precision[rank] = round(precisionCounter / rank, 4)
                avgPrecisionArr.append(precision[rank])

                recall[rank] = round(recallCounter / relcount, 4)

                data = [str(rank), "R", str(precision[rank]), str(recall[rank])]
                formatStr = '{0[0]:<15}{0[1]:<15}{0[2]:<5}{0[3]:>15}'.format(data)
                file.write(formatStr+"\n")

            else:
                precision[rank] = round(precisionCounter/rank, 4)
                recall[rank] = round(recallCounter/relcount, 4)

                data = [str(rank), "N", str(precision[rank]), str(recall[rank])]
                formatStr = '{0[0]:<15}{0[1]:<15}{0[2]:<5}{0[3]:>15}'.format(data)
                file.write(formatStr + "\n")

        else:
            return precision, recall


    # calculate Average precision
    avgPrecisionSum = 0
    for val in avgPrecisionArr:
        avgPrecisionSum += val

    if len(avgPrecisionArr) > 0:
        avgPrecision = avgPrecisionSum/len(avgPrecisionArr)
    else:
        avgPrecision = 0

    precision["AP"] = avgPrecision

    file.write("**********************************\n\n")
    file.write("Average Precision -> "+ str(avgPrecision)+"\n")

    file.write("Precision at K = 5 :"+ str(precision[5]) + "\n")
    file.write("Precision at K = 20 :"+ str(precision[20]) + "\n")
    file.flush()
    file.close()


    return precision, recall

main()