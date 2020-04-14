import nltk
import math 
import operator
import numpy as np
from pickle import dump, load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer,sent_tokenize

'''
+-----------------------------------------------------------------------+
|                                                                       |
|            This program obtains the context of one word               |
|            and found the words similarity it. Using the               |
|              Syntagmatic Relations of the words and                   |
|                        Conditional  Entropy                           |
|                                                                       |
+-----------------------------------------------------------------------+
'''

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    return text

def openText():
    f = open('../../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()
    return text

def normalization(Sentences):
    sent=[]
    for s in Sentences:
        aux=tokens(s)
        sent.append(aux)
    return sent    

def tokens(text):
    stop=stopwords.words('spanish')
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    words=[w for w in words if w.lower() not in stop]
    words= " ".join(words)
    return words

def tagSentences(Sentences,combined_tagger):
    Stag=[]
    for s in Sentences:
        aux=s.split(' ')
        s_tagged=combined_tagger.tag(aux)
        aux=cleanTagger(s_tagged)
        tag=' '.join(aux)
        Stag.append(tag)
    return Stag

def lemaSentences(Sentences,lemmas):
    sentencesL=[]
    for s in Sentences:
        aux=s.split(' ')
        newSentences=[]
        for word in aux:
            if word in lemmas:
                newSentences.append(lemmas[word])
            else:
                newSentences.append(word)
        sentencesL.append(' '.join(newSentences))
    return sentencesL

def getSentences(text):
    text=cleaningText(text)
    sentences=sent_tokenize(text)
    
    return sentences

def getVocabulary(text):
    stop=stopwords.words('spanish')
    t=cleaningText(text)
    t=nltk.word_tokenize(t)
    t=[w.lower() for w in t if w.isalpha()]
    t=[w for w in t if w.lower() not in stop]
    return t

def getProbability(word,sentences):
    suma=0
    for s in sentences:
        if(bool(s.count(word))):
            suma=suma+1
    return (suma+0.5) / ( len(sentences) + 1 )

def getProbability2(word1,word2,sentences):
    suma=0
    for s in sentences:
        if(bool(s.count(word1) and bool(s.count(word2)))):
            suma=suma+1
    return (suma + 0.25) / ( len(sentences) + 1 )

'''
-----------------------------------------------------------
|             |  P1     p(w2=0)      |  P2   p(w2=1)      |
-----------------------------------------------------------
| P3 p(W1=0)  |  P4   p(w1=0|w2=0)   |  P5  p(w1=0|w2=1)  |
-----------------------------------------------------------
| P6 p(W1=1)  |  P7   p(w1=1|w2=0)   |  P8  p(w1=1|w2=1)  |
-----------------------------------------------------------

'''
def getTableProbability( pWord1 , pWord2 , pW1AndpW2 ):
    p5 = pWord2 - pW1AndpW2
    p1 = 1 - pWord2
    p7 = pWord1 - pW1AndpW2
    p4 = p1 - p7
    p3 = p4 + p5
    table = [ p1 , pWord2 , p3 , p4 , p5 , pWord1 , p7 , pW1AndpW2 ]
    return table  

def getEntropy( table ):
    x1 = table[3] * math.log2( table[3] / table[0] ) + table[6] * math.log2( table[6] / table[0] )
    x2 = table[4] * math.log2( table[4] / table[1] ) + table[7] * math.log2( table[7] / table[1] )
    H = -1 * ( x1 + x2 )
    return H 

def lemmetization(tokens,lemmas):
    newTokens=[]
    for token in tokens:
        if token in lemmas:
            newTokens.append(lemmas[token])
        else:
            newTokens.append(token)
    return newTokens

def getGenerate():
    fopen="../../Corpus/generate.txt"
    archivo= open(fopen,encoding='utf-8')
    lemmas={}
    for linea in archivo.readlines(): 
        lemmas[linea.split(' ')[0].replace("#","")]=linea.split(' ')[-1][:-1]
    archivo.close()
    return lemmas

def generateTagger():      
    default_tagger=nltk.DefaultTagger('V')
    patterns=[  (r'.*o$', 'NMS'), # noun masculine singular
                (r'.*os$', 'NMP'), # noun masculine plural
                (r'.*a$', 'NFS'),  # noun feminine singular
                (r'.*as$', 'NFP')  # noun feminine plural
            ]
    regexp_tagger=nltk.RegexpTagger(patterns, backoff=default_tagger)
    cess_tagged_sents=cess_esp.tagged_sents()
    combined_tagger=nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    return combined_tagger

def cleanTagger(s_tagged):
    list(s_tagged)
    vocabulary=[]
    for i in range(len(s_tagged)):
        vocabulary.append(s_tagged[i][0]+" "+s_tagged[i][1])
    
    return vocabulary

def pkl(f,info):
    output=open(f, 'wb')
    dump(info, output, -1)
    output.close()
    
def getPKL(fileName): 
    with open(fileName,'rb') as f:
        return load(f)
        
if __name__ == "__main__":
    #**********************************************************************************
    #                      Run the first time for generate the files .pkl
    #**********************************************************************************
    text=openText()
    #Vocabulario con lemmas
    print("get vocabulary")
    lemmas = getGenerate()
    stop = stopwords.words('spanish')
    
    vocabulary= getVocabulary(text)
    vocabulary=lemmetization(vocabulary,lemmas)
    s_tagged=generateTagger(vocabulary)
    vocabulary=cleanTagger(s_tagged)
    vocabulary=sorted(set(vocabulary))
    pkl('vocabulary.pkl',vocabulary)
    
    sentences=getSentences(text)
    pkl('Sentences1.pkl',sentences)

    print("Normalization")
    CleanSentences=normalization(sentences)
    pkl('Sentences2.pkl',CleanSentences)
    print("Lemmas")
    lemmasSent=lemaSentences(CleanSentences,lemmas)
    pkl('Sentences3.pkl',lemmasSent)
    print("Tag")
    print(lemmasSent)
    combined_tagger=generateTagger()
    tagSent=tagSentences(lemmasSent,combined_tagger)
    pkl('Sentences4.pkl',tagSent)

    #**********************************************************************************
    #                                    Load the files .pkl
    #**********************************************************************************

    #vocabulary = getPKL('vocabulary.pkl')
    #sentences = getPKL('Sentences1.pkl')
    #CleanSentences = getPKL('Sentences2.pkl')
    #lemmasSent = getPKL('Sentences3.pkl')
    #tagSent = getPKL('Sentences4.pkl')

    #word1='grande aq0cs0'
    #word1='abastecer V'
    word1 = 'economía ncfs000'
    #word1='nacional aq0cs0'

    entropy = [ ]

    pWord1= getProbability( word1,tagSent )
    for termn in vocabulary:
        pWord2 = getProbability( termn,tagSent )
        pW1AndpW2 =  getProbability2( word1 , termn , tagSent )
        table = getTableProbability( pWord1 , pWord2 , pW1AndpW2 ) 
        H = getEntropy( table )
        entropy.append( (termn,H) )

    entropy = sorted(entropy,key=operator.itemgetter(1))

    fv=open('Entropy_'+ word1.split(' ')[0] +'.txt','w')
    for e in entropy:
        fv.write( '{:30}{:30}\n'.format(e[0],e[1]) )

