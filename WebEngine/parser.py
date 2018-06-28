#Om Sai Ram
import re
import requests as req
from bs4 import BeautifulSoup as bs
import string
import sys
import os
import glob
import html2text
import nltk


def processEachFile(filename):

    file = open(filename, "r")

    h = html2text.HTML2Text()
    h.ignore_emphasis = True
    h.ignore_images = True
    h.ignore_links = True
    h.ignore_tables = True

    txt = h.handle(file.read())
    print("Befpre Process")
    print(txt)

    regex = r"(?<!\d)[.,;:](?!\d)"

    pattern = r'''(?x)
                    (?:[A-Z]\.)+
                    | \w+(?:-\w)*
                    | \$?\d+(?:\.\d+)?%? '''


    # if filename == './cacm/CACM-3149.html':
    #     print("here")


    allText = txt.lower()

    indx = allText.rfind("pm")
    if indx >= 0 :
        allText = allText[: (indx + 2)]
    else:
        indx = allText.rfind("am")
        allText = allText[: (indx + 2)]


    allwords = allText.split()

    tokenLst = []
    for word in allwords:
        # checking if this is only digits then do so ie; no text
        if(re.search('[a-zA-Z]', word) == None):
            result = re.sub(regex, "", word, 0)
            for char in ['{', '}', '(', ')', '[', ']']:
                result = str.replace(result, char, "")

            if result.startswith(","):
                result = result[1:]
            if result.endswith(","):
                result = result[:-1]
            if result.isspace():
                continue
            if not result:
                continue
            tokenLst.append(result)
        # this condition is for hyphenated words
        elif "-" in word:
            tokenLst.append(word)
        # All other words , do this
        else:
            tokenize_words = nltk.regexp_tokenize(word, pattern)
            tokenLst += tokenize_words

    print("After Process")
    print(tokenLst)

    file.close()

    return tokenLst

def main():
    defaultInputFolder = './cacm'

    if len(sys.argv) == 2:
        if sys.argv[1]:
            defaultInputFolder = sys.argv[1]

    count = 1

    for filename in glob.glob(os.path.join(defaultInputFolder, '*.html')):

        #Get tokens from the HTML file
        tokens = processEachFile(filename)

        #getting DOCID or main doc name
        segments = filename.rpartition('/')
        filenameKey = segments[len(segments) - 1]
        subparts = filenameKey.split('.')
        filenameKey = subparts[0]

        #writing tokens to file and flushing buffer for immediate write
        file = open("./tokens/"+filenameKey+".txt", "w")
        for word in tokens:
            file.write(word)
            file.write("\n")

        file.flush()
        file.close()

        print(count)
        count += 1

    return

main()

# # word = "Newton-Raphson"
# word = "7:8)[{("
#
# print(word.isdigit())
# print(word.isalnum())
# print(re.search('[a-zA-Z]', word)== None)
#
# regex = r"(?<!\d)[.,;:](?!\d)"
#
# result = re.sub(regex, "", word, 0)
# print(result)
#
#
# pattern = r'''(?x)
#     (?:[A-Z]\.)+
#     | \w+(?:-\w)*
#     | \$?\d+(?:\.\d+)?%? '''
#
#
# tokenize_words = nltk.regexp_tokenize(word, pattern)
#
# print(tokenize_words)
#
#
# translator = str.maketrans("", "", string.punctuation)
#
# print(word.translate(translator))
#
#
# pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
# tokenizer = nltk.RegexpTokenizer(pattern)
# print(tokenizer.tokenize(word))
#
# for char in ['{','}','(',')','[',']']:
#     word = str.replace(word,char,"")
# print(word)





