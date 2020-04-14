import nltk
import math 
import numpy as np
import operator
from pickle import dump, load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|            and found the words similarity it. Using the               |
|             Syntagmatic Relations of the words and                    |
|                                BM25                                   |
|                                                                       |
+-----------------------------------------------------------------------+
'''

def openText():
    f = open('../../Corpus/e961024.htm', encoding = 'utf-8')
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

def normalization(text):
    stop = stopwords.words('spanish')
    t = cleaningText(text)
    t = tokens(t)
    t = [w for w in t if w.lower() not in stop]
    return t

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    return text

def tokens(text):
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    return words

def lemmatized( token, lemmas):
    tokensLemmatized = []
    for tok in token:
        try:
            tokensLemmatized.append(lemmas[tok])
        except KeyError:
            tokensLemmatized.append(tok)
    return tokensLemmatized 

def generateTagger(clean_vocabulary):      
    fname = 'combined_taggerP.pkl'
    default_tagger = nltk.DefaultTagger('V')
    patterns=[  (r'.*o$', 'NMS'), # noun masculine singular
                (r'.*os$', 'NMP'), # noun masculine plural
                (r'.*a$', 'NFS'),  # noun feminine singular
                (r'.*as$', 'NFP')  # noun feminine plural
            ]
    regexp_tagger = nltk.RegexpTagger(patterns, backoff=default_tagger)
    cess_tagged_sents = cess_esp.tagged_sents()
    combined_tagger = nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    s_tagged = combined_tagger.tag(clean_vocabulary)
    output = open(fname, 'wb')
    dump(s_tagged, output, -1)
    output.close()

    return s_tagged

def cleanTagger(s_tagged):
    list(s_tagged)
    vocabulary = []
    for i in range(len(s_tagged)):
        vocabulary.append(s_tagged[i][0]+' '+s_tagged[i][1])

    return vocabulary

def getContext(vocabulary,cleanWords):
    contexts = {}
    window = 4
    for termino in vocabulary:
        context = []
        for j,word in enumerate(cleanWords):
            if termino == word :
                context += cleanWords[ 0 if j-window < 0 else j-window : j ] 
                context += cleanWords[ j+1 : j+(window+1) if j < len(cleanWords) else len(cleanWords) ] 
        contexts[termino] = context
    output = open('contextT.pkl', 'wb')
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
    
    output=open('vectoresT.pkl', 'wb')
    dump(vectors, output, -1)
    output.close()
    return vectors

def getPromContexts( contexts ):
    avdl = 0
    for key in contexts:
        avdl += len(contexts[key])
    return avdl / 6194

def getBM25( vectors , avdl):
    k = 1.2
    b = 0.75
    vectorsBM25 = {}
    for key in vectors:
        frec = vectors[key]
        value = np.array(frec)
        d1 = np.sum(value)
        vectorFrecuency = []

        for f in frec:
            tf = ( ( k+1 )*f ) / ( f + k * ( ( 1-b + b * d1 ) / avdl ) )
            vectorFrecuency.append(tf)
        vectorsBM25[key] = vectorFrecuency
    return vectorsBM25

def normalizationBM25( vectorsBM25 ):
    vectors = {}
    for key in vectorsBM25:
        vector = vectorsBM25[key]
        value = np.array(vector)
        prom = np.sum(value)
        newVector = []
        for v in vector:
            newVector.append(v/prom)
        vectors[key] = newVector
    return vectors

def getFrec(contexts):
    doc_frec = []
    times = 0
    for word in contexts:
        for key in contexts:
            context = contexts[key]
            if( word in context ):
                times = times+1
        doc_frec.append(times) 
    return doc_frec

def getIDF(doc_frec):
    v_idf = []
    for i in enumerate(doc_frec):
        val = math.log((6194+1)/doc_frec[i[0]])
        v_idf.append(val)
    return v_idf

def getTF_IDF(v_tf,v_idf):
    v = {}
    idf = np.array(v_idf)
    for key in v_tf:
        vtf = v_tf[key]
        mul = np.multiply(vtf,idf)
        v[key] =  mul
    return v

def getRelations( tf_idf , word , finalVocabulary ):
    relation = []
    vector = tf_idf[word]
    for i,v in enumerate(vector):
        relation.append( ( finalVocabulary[i] , v ) )
    relation = sorted(relation,key=operator.itemgetter(1),reverse=True)
    return relation

def getCosines(vocabulary,tf_idf,word):
    cosines = {}
    value = tf_idf[word]
    value = np.array(value)
    vector = []
    for voc in vocabulary:
        vector = tf_idf[voc]
        vector = np.array(vector)
        cosine = np.dot(value,vector)/((np.sqrt(np.sum(value**2)))*(np.sqrt(np.sum(vector**2))))
        cosines[voc] = cosine
    output = open('cosinesT.pkl', 'wb')
    dump(cosines, output, -1)
    output.close()

    return cosines

def getPickle(fileName): 
    with open(fileName,'rb') as f:
        return load(f)


if __name__ == '__main__':
    text = openText()
    tokens = normalization( text )
    lemmas = openGenerate()
    tokenLemma = lemmatized( tokens , lemmas )
    s_tagged = generateTagger( tokenLemma )
    tokensLemmas = cleanTagger( s_tagged )
    finalVocabulary = sorted( set(tokensLemmas) )
    print(len(tokensLemmas))
    print(len(finalVocabulary))
    print('get context')
    context = getContext(finalVocabulary,tokensLemmas)
    #context = getPickle('contextT.pkl')
    print('get vector')
    vectors = getVector(finalVocabulary,context)
    #vectors = getPickle('vectoresT.pkl')
    print('get Promedio')
    avdl = getPromContexts(context)
    print(avdl)
    print('get vector BM25')
    vectorsBM25 = getBM25( vectors,avdl ) 
    print('Normalizando BM25')
    vectorsBM25 = normalizationBM25( vectorsBM25 )
    print('get Doc frec')
    doc_frec = getFrec(context)
    print('get vector idf') 
    v_idf = getIDF(doc_frec)
    tf_idf = getTF_IDF( vectorsBM25 , v_idf)
    word='grande aq0cs0'
    relations = getRelations( tf_idf , word , finalVocabulary )
    print('imprimir')
    fv=open('similitudLemmas.txt','w')
    for r in relations:
        fv.write('{:20}{:25}\n'.format(r[0],r[1]))