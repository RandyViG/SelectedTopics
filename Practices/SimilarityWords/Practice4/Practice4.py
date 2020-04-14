import nltk
import operator
import numpy as np
from pickle import dump,load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|            and found the words similarity it. Using the               |
|                          Stems of the words                           |
|                                                                       |
+-----------------------------------------------------------------------+
'''

def openText():
    f = open('../../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()
    return text

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

def stemming(tokens):
    newTokens=[]
    fopen="../../Corpus/generate.txt"
    archivo= open(fopen,encoding='utf-8')
    lemmas={}
    for linea in archivo.readlines(): 
        lemmas[linea.split(' ')[0].replace("#","")]=linea.split(" ")[0][:linea.split(" ")[0].find("#")]
    archivo.close()
    
    for token in tokens:
        if token in lemmas:
            newTokens.append(lemmas[token])
        else:
            newTokens.append(token)
    return newTokens

def generateTagger(clean_vocabulary):      
    fname= 'combined_tagger.pkl'
    default_tagger=nltk.DefaultTagger('V')
    patterns=[  (r'.*o$', 'NMS'), # noun masculine singular
                (r'.*os$', 'NMP'), # noun masculine plural
                (r'.*a$', 'NFS'),  # noun feminine singular
                (r'.*as$', 'NFP')  # noun feminine plural
            ]
    regexp_tagger=nltk.RegexpTagger(patterns, backoff=default_tagger)
    #train nltk.UnigramTagger using tagged sentences from cess_esp 
    cess_tagged_sents=cess_esp.tagged_sents()
    combined_tagger=nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    #save the trained tagger in a file
    output=open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()
    s_tagged=combined_tagger.tag(clean_vocabulary)
    return s_tagged

def cleanTagger(s_tagged):
    list(set(s_tagged))
    vocabulary=[]
    for i in range(len(s_tagged)):
        vocabulary.append(s_tagged[i][0]+" "+s_tagged[i][1])
    return vocabulary
    
def getContext(vocabulary,cleanWords):
    windowSize = 4
    contexts = {}
    for termino in vocabulary:
        context = []
        for j,word in enumerate(cleanWords):
            if termino == word :
                context+=cleanWords[ 0 if j-windowSize < 0 else j-windowSize : j ] 
                context+= cleanWords[ j+1 : j+windowSize+1 if j < len(cleanWords) else len(cleanWords) ] 
        contexts[termino] = context
    return contexts

def getVector(vocabulary,contexts):
    vectors={}
    for termino in vocabulary:
        context=contexts[termino]
        vector=[]
        for word in vocabulary:
            frec=context.count( word )
            vector.append( frec )
        vectors[termino]=vector
    return vectors

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
    return cosines

def writeFile(word,cosines):
    f= open('similar'+word+'.txt','w')
    for item in cosines:
        word=item[0]
        cosine=item[1]
        string = str(word) + ' ' + str(cosine) +'\n'
        f.write(string)
    f.close()

if __name__ == "__main__":
    text=openText()
    tokens = normalization( text )
    tokensTerm = stemming( tokens )
    tokenTagged = generateTagger( tokensTerm )
    s_tagged = cleanTagger( tokenTagged )
    vocabulary = sorted( set(s_tagged) )
    print("get context")
    context = getContext( vocabulary , s_tagged )
    print("get vector")
    vectors = getVector( vocabulary , context )
    word="grande aq0cs0"
    print("get cosines")
    vectorCosines = getCosines( vocabulary , vectors , word )
    print("get ordenando")
    vectorCosines = sorted(vectorCosines.items(),key=lambda kv: kv[1],reverse=True)
    print("imprimir")
    fv=open("similitudes.txt","w")
    for key in vectorCosines:
        if (key[0].split(" ")[1][0] == 'a'):
            fv.write( '{:20} {:15} \n'.format( key[0].split(' ')[0] + ' ' +key[0].split(' ')[1][0] , key[1] ) )
    fv.close()
    