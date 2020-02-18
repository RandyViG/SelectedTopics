import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import operator

#Lectura
def openText():
    f = open('Corpus/e961024.htm', encoding='utf-8')
    text = f.read()
    f.close()
    return text

#Limpiar HTML
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
    #windowsSize = 8
    contexts = {}
    for termino in vocabulary:
        context = []
        for j,word in enumerate( tokens ):
            if termino == word :
                context += tokens[ 0 if j-4 < 0 else j-4 : j ]
                context += tokens[ j+1 : j+5 if j < len(tokens) else len(tokens) ] 
        #print(context)
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
        string = str(word)+str(cosine)+'\n'
        f.write(string)
    f.close()


if __name__ == "__main__":
    stop = stopwords.words('spanish')
    text = openText()
    cleanText = cleanHtml( text )
    tokens = tokenize( cleanText )
    t = [w for w in tokens if w.lower() not in stop]
    vocabulary = sorted(set( t ))
    #vocabulary = [w for w in sortedWords if w.lower() not in stop]
    contexts = getContext(vocabulary, t)
    vectors = getVector(vocabulary,contexts)
    word='grande'
    cosines = getCosines(vocabulary,vectors,word)
    cosines_sorted = sorted(cosines.items(),key=operator.itemgetter(1),reverse=True)
    writeFile(word,cosines_sorted)
    #for term in enumerate(cosines_sorted):
    #    print( term[1][0] , cosines[ term[1][0] ] )
    

     
'''
for w in vocabulary:
    for i in range( len ( vocabulary ):
        if vocabulary[i] == w:
            for j in range(i-int(windowsSize/2),i):
                if j>=0:
                    context.append(vocabulary[j])
            try:
                for j in range (i+1, i+(int(windowsSize/2+1)) ):
                    context.append(vocabulary[j])
            except IndexError:
                pass
            contexts[w] = context
'''
