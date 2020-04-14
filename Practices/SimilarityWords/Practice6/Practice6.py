import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from pickle import dump, load
import numpy as np
import operator

'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|            and found the words similarity it. Using the               |
|             Stems of the words with nltk.stem Snowball                |
|                 Stemmer and normalizes the frequency                  |
|                            (Probability)                              |
|                                                                       |
+-----------------------------------------------------------------------+
'''

def openText():
    f = open('../../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()
    return text

def normalization(text):
    stop=stopwords.words('spanish')
    t=cleaningText(text)
    t=tokens(t)
    t=[w for w in t if w.lower() not in stop]
    vocabulary=sorted(set(t))
    return vocabulary,t

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    return text

def tokens(text):
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    return words

def getContext(vocabulary,cleanWords):
    contexts = {}
    window = 4 
    for termino in vocabulary:
        context = []
        for j,word in enumerate(cleanWords):
            if termino == word :
                context+=cleanWords[ 0 if j-window < 0 else j-window : j ] 
                context+= cleanWords[ j+1 : j+window+1 if j < len(cleanWords) else len(cleanWords) ] 
        contexts[termino] = context

    output=open("context.pkl", 'wb')
    dump(contexts, output, -1)
    output.close()
    
    return contexts

def getVector(vocabulary,contexts):
    vectors={}
    for termino in vocabulary:
        context=contexts[termino]
        vector=[]
        for word in vocabulary:
            frec=context.count(word)
            vector.append(frec)
        vectors[termino]=vector
    output=open("vectores.pkl", 'wb')
    dump(vectors, output, -1)
    output.close()
    return vectors

def getProbability( finalVocabulary ,vectors ):
    probabilities = {}
    for termino in finalVocabulary:
        proba = [ ]
        vec = vectors[termino]
        value=np.array( vec )
        suma = np.sum(value)
        for v in vec:
            proba.append(v/suma)
        probabilities[termino] = proba
    return probabilities

def getCosines(vocabulary,vectors,word):
    cosines={}
    value=vectors[word]
    value=np.array(value)
    vector=[]
    for voc in vocabulary:
        vector=vectors[voc]
        vector=np.array(vector)
        cosine=np.dot(value,vector)/((np.sqrt(np.sum(value**2)))*(np.sqrt(np.sum(vector**2))))
        cosines[voc]=cosine
    output=open("cosines.pkl", 'wb')
    dump(cosines, output, -1)
    output.close()
    return cosines

def generateTagger(clean_vocabulary):      
    fname= 'combined_tagger.pkl'
    default_tagger=nltk.DefaultTagger('V')
    patterns=[  (r'.*o$', 'NMS'), # noun masculine singular
                (r'.*os$', 'NMP'), # noun masculine plural
                (r'.*a$', 'NFS'),  # noun feminine singular
                (r'.*as$', 'NFP')  # noun feminine plural
            ]
    regexp_tagger=nltk.RegexpTagger(patterns, backoff=default_tagger)
    cess_tagged_sents=cess_esp.tagged_sents()
    combined_tagger=nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    s_tagged=combined_tagger.tag(clean_vocabulary)
    output=open(fname, 'wb')
    dump(s_tagged, output, -1)
    output.close()

    return s_tagged

def cleanTagger(s_tagged):
    list(s_tagged)
    vocabulary=[]
    for i in range(len(s_tagged)):
        vocabulary.append(s_tagged[i][0]+" "+s_tagged[i][1])

    return vocabulary

def Stemm(tokens):
    newTokens=[]
    ss = SnowballStemmer('spanish')
    for token in tokens:
        newTokens.append(ss.stem(token))
    return newTokens

def getPickle(fileName): 
    with open(fileName,'rb') as f:
        return load(f)


if __name__ == "__main__":
    text=openText()
    vocabulary,tokens=normalization(text)
    tokenStemmer = Stemm(tokens)
    s_tagged = generateTagger(tokenStemmer)
    tokenStemmer = cleanTagger(s_tagged)
    finalVocabulary = sorted(set(tokenStemmer))
    print("get context")
    context = getContext(finalVocabulary,tokenStemmer)
    print("get vector")
    vectors = getVector(finalVocabulary,context)
    word = "gran aq0cs0"
    probabilities = getProbability( finalVocabulary , vectors )
    print("get cosines")
    vectorCosines = getCosines(finalVocabulary,probabilities,word)
    print("get ordenando")
    vectorCosines = sorted(vectorCosines.items(),key=lambda kv: kv[1],reverse=True)
    
    print("imprimir")
    fv=open("similitud.txt","w")
    for key in vectorCosines:
        if (key[0].split(" ")[1][0] == 'a'): 
            fv.write( '{:20} {:15} \n'.format( key[0].split(' ')[0] + ' ' +key[0].split(' ')[1][0] , key[1] ) )
    fv.close()