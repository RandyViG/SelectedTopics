import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import operator
'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|                   and found the words similarity it.                  |
|                                                                       |
+-----------------------------------------------------------------------+
'''
#Read
def openText():
    f = open('../../Corpus/e961024.htm', encoding='utf-8')
    text = f.read()
    f.close()
    return text

#Clean HTML
def cleanHtml( text ):
    soup = BeautifulSoup(text,'lxml')
    cleanText = soup.get_text()
    return cleanText

#Tokenizar
def tokenize( text ):
    words = nltk.word_tokenize(text)
    lowerWords = [ word.lower() for word in words if word.isalpha() ]
    return lowerWords 

#Concordance
def getContext( vocabulary, tokens ):
    windowSize = 4
    contexts = {}
    for termino in vocabulary:
        context = []
        for j,word in enumerate( tokens ):
            if termino == word :
                context += tokens[ 0 if j-windowSize < 0 else j-windowSize : j ]
                context += tokens[ j+1 : j+windowSize+1 if j < len(tokens) else len(tokens) ] 
        contexts[termino] = context
    return contexts

def getVector( vocabulary,contexts ):
    vectors = {}
    for termino in vocabulary:
        vector=[]
        context = contexts[termino]
        for word in vocabulary:
            frec = context.count(word)
            vector.append(frec)
        vectors[ termino ] = vector
    return vectors
        
def getCosines( vocabulary,vectors,word):
    cosines = {}
    value = vectors[word]
    value = np.array(value)
    for voc in vocabulary:
        vector = vectors[voc]
        vector = np.array(vector)
        cosine = np.dot(value,vector) / ( (np.sqrt(np.sum(value**2))) * (np.sqrt(np.sum(vector**2))) )
        cosines[voc] = cosine
    return cosines

def writeFile(word,cosines):
    f = open('WordsSimilarTo'+word+'.txt','w')
    for item in cosines:
        word = item[0]
        cosine = item[1]
        f.write('{:20}  {:30}\n'.format( str(word) , str(cosine) ) )
    f.close()

if __name__ == "__main__":
    stop = stopwords.words('spanish')
    text = openText()
    cleanText = cleanHtml( text )
    tokens = tokenize( cleanText )
    t = [w for w in tokens if w.lower() not in stop]
    vocabulary = sorted(set( t ))
    contexts = getContext(vocabulary, t)
    vectors = getVector(vocabulary,contexts)
    word='grande'
    cosines = getCosines(vocabulary,vectors,word)
    cosines_sorted = sorted(cosines.items(),key=operator.itemgetter(1),reverse=True)
    writeFile(word,cosines_sorted)