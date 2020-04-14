import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import operator
from pickle import dump, load
from nltk.corpus import cess_esp

'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|            and found the words similarity it. Using the               |
|                          Lemma of the words                           |
|                                                                       |
+-----------------------------------------------------------------------+
'''

#Lectura
def openText():
    f = open('../../Corpus/e961024.htm', encoding='utf-8')
    text = f.read()
    f.close()
    return text

def openGenerate():
    f = open('../../Corpus/generate.txt',encoding='utf-8')
    lemmas = {}
    for line in f.readlines():
        lemmas[ line.split(' ')[0].replace('#','') ] = line.split(' ')[-1][:-1]
    f.close()
    return lemmas

def normalization( text ):
    stop = stopwords.words('spanish')
    cleanText = cleanHtml( text )
    tokens = tokenize( cleanText )
    t = [w for w in tokens if w.lower() not in stop]
    return t
    
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

#Context
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

    output = open('context.pkl','wb')
    dump(contexts,output,-1)
    output.close()

    return contexts

def getVector( vocabulary,contexts ):
    vectors = {}
    for termino in vocabulary:
        vector=[]
        context = contexts[termino]
        for word in vocabulary:
            frec = context.count( word )
            vector.append( frec )
        vectors[ termino ] = vector

    output = open('vectors.pkl','wb')
    dump(vectors,output,-1)
    output.close()
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
    output = open('cosines.pkl','wb')
    dump(vectors,output,-1)
    output.close()
    return cosines

def lemmatized( token, lemmas):
    tokensLemmatized = []
    for tok in token:
        try:
            tokensLemmatized.append(lemmas[tok])
        except KeyError:
            tokensLemmatized.append(tok)
    return tokensLemmatized 

def loadPickle( fileName ):
    with open (fileName,'rb') as f:
        return load(f)


def generateTagger( tokens ):
    fname = 'combined_tagger.pkl'
    default_tagger = nltk.DefaultTagger('V')
    patterns=[ (r'.*o$', 'NMS'), # noun masculine singular
            (r'.*os$', 'NMP'), # noun masculine plural
            (r'.*a$', 'NFS'),  # noun feminine singular
            (r'.*as$', 'NFP')  # noun feminine singular
            ]
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    #train nltk.UnigramTagger using tagged sentences from cess_esp 
    cess_tagged_sents = cess_esp.tagged_sents()
    combined_tagger = nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    s_tagged = combined_tagger.tag( tokens )
    #save the trained tagger in a file
    output = open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()
    
    return s_tagged

def cleanTagged( s_tagged ):
    list( set( s_tagged ))
    finalVocabulary=[]
    for i in range(len(s_tagged)):
        finalVocabulary.append(s_tagged[i][0]+" "+s_tagged[i][1])

    return finalVocabulary


if __name__ == "__main__":
    #**********************************************************************************
    #                      Run the first time for generate the files .pkl
    #**********************************************************************************
    text = openText()
    lemmas = openGenerate()
    tokens = normalization( text )
    tokenLemmatized = lemmatized( tokens , lemmas )
    tagged = generateTagger ( tokenLemmatized )
    tokenTagged = cleanTagged( tagged )
    vocabulary = sorted( set( tokenTagged ) )
    contexts = getContext( vocabulary , tokenTagged )
    vectors = getVector( vocabulary, contexts )
    word = 'grande aq0cs0'
    vectorCosines = getCosines(vocabulary,vectors,word)

    #**********************************************************************************
    #                                    Load the files .pkl
    #**********************************************************************************
    #contexts = loadPickle('context.pkl') 
    #vectors = loadPickle('vectores.pkl')
    #vectorCosines = loadPickle('cosines.pkl')
    
    vectorCosines = sorted(vectorCosines.items(), key = lambda kv: kv[1], reverse = True)
    fv = open("similitudes.txt","w")

    print('Guardando\n')
    for key in vectorCosines:
        if ( key[0].split(' ')[1][0] == 'a' ):
            fv.write( '{:20} {:15} \n'.format( key[0].split(' ')[0] + ' ' +key[0].split(' ')[1][0] , key[1] ) )
    fv.close()