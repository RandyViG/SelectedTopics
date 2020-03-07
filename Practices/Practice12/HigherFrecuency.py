import nltk
import math 
import operator
from pickle import dump, load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def openText():
    f = open('../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()

    return text

def openGenerate():
    f = open('../generate.txt',encoding='utf-8')
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
    vocabulary = sorted(set(t))

    return vocabulary,t

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

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    
    return text

def tokens(text):
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    
    return words

def nounVocabulary( vocabulary ):
    nounVocabulary = [ ]
    for term in vocabulary:
        if term.split(' ')[1][0] == 'n':
            nounVocabulary.append(term)

    return nounVocabulary

def getTopic( vocabulary , tokens ):
    topics = [ ]
    for term in vocabulary:
        frecuency = tokens.count( term )
        topics.append( (term,frecuency) )
    
    return topics

def getPickle(fileName): 
    with open(fileName,'rb') as f:
        return load(f)


if __name__ == '__main__':
      
    text = openText()
    vocabulary,tokens = normalization( text )
    lemmas = openGenerate()
    tokenLemma = lemmatized( tokens , lemmas )
    s_tagged = generateTagger( tokenLemma )
    tokensLemmas = cleanTagger( s_tagged )
    
    finalVocabulary = sorted( set(tokensLemmas) )
    finalVocabulary = nounVocabulary( finalVocabulary )
    
    print(len(vocabulary))

    topics = getTopic( finalVocabulary , tokensLemmas )
    topics = sorted( topics,key=operator.itemgetter(1),reverse=True )

    fv=open('HigherFrecuency.txt','w')
    for t in topics:
        fv.write( '{:30}{:30}\n'.format(t[0],str(t[1])) )
    fv.close()          
    #5172