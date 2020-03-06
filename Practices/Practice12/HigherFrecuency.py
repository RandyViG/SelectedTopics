import nltk
import operator
from pickle import dump, load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer,sent_tokenize

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()

    return text

def openText():
    f = open('../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()

    return text

def getGenerate():
    fopen="../generate.txt"
    archivo= open(fopen,encoding='utf-8')
    lemmas={}
    for linea in archivo.readlines(): 
        lemmas[linea.split(' ')[0].replace("#","")]=linea.split(' ')[-1][:-1]
    archivo.close()

    return lemmas

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

def getVocabulary(text):
    stop=stopwords.words('spanish')
    t=cleaningText(text)
    t=nltk.word_tokenize(t)
    t=[w.lower() for w in t if w.isalpha()]
    t=[w for w in t if w.lower() not in stop]

    return t 

def lemmetization(tokens,lemmas):
    newTokens=[]
    for token in tokens:
        if token in lemmas:
            newTokens.append(lemmas[token])
        else:
            newTokens.append(token)

    return newTokens

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

def cleanVocabulary( vocabulary ):
    nounVocabulary = [ ]
    for term in vocabulary:
        if term.split(' ')[1][0] == 'n':
            nounVocabulary.append(term)

    return nounVocabulary

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

def pkl(f,info):
    output=open(f, 'wb')
    dump(info, output, -1)
    output.close()
    
def getPKL(fileName): 
    with open(fileName,'rb') as f:
        return load(f)

def getTopics( vocabulary , sentences ):
    topics = [ ]
    frecuency = 0.0
    for term in vocabulary:
        for s in sentences:
            frecuency += s.count(term)
        topics.append( ( term , frecuency ) )
        
    return topics
        
if __name__ == "__main__":
    text=openText()
    #Vocabulario con lemmas
    print("get vocabulary")
    lemmas=getGenerate()
    stop=stopwords.words('spanish')
    '''
    vocabulary= getVocabulary(text)
    vocabulary=lemmetization(vocabulary,lemmas)
    s_tagged=generateTagger(vocabulary)
    vocabulary=cleanTagger(s_tagged)
    vocabulary=sorted(set(vocabulary))
    
    
    pkl('vocabulary.pkl',vocabulary)
    vocabulary=getPKL('vocabulary.pkl')
    
    nounVocabulary = cleanVocabulary( vocabulary )
    pkl('nounVocabulary.pkl',nounVocabulary)
    
    '''

    vocabulary = getPKL('nounVocabulary.pkl')


    print("get sentences")
    '''
    #sentences = getSentences(text)
    #pkl('Sentences1.pkl',sentences)
    sentences=getPKL('Sentences1.pkl')
    print("\tNormalization")
    #CleanSentences = normalization(sentences)
    #pkl('Sentences2.pkl',CleanSentences)
    CleanSentences = getPKL('Sentences2.pkl')
    print("\tLemmas")
    #lemmasSent = lemaSentences(CleanSentences,lemmas)
    #pkl('Sentences3.pkl',lemmasSent)
    lemmasSent = getPKL('Sentences3.pkl')
    print("\tTag")
    #print(lemmasSent)
    #combined_tagger = generateTagger()
    #tagSent = tagSentences(lemmasSent,combined_tagger)
    #pkl('Sentences4.pkl',tagSent)
    '''
    tagSent=getPKL('Sentences4.pkl')

    topics = getTopics( vocabulary , tagSent )

    topics = sorted(topics,key=operator.itemgetter(1),reverse=True)

    fv=open('HigherFrequencyTopic.txt','w')
    for t in topics:
        fv.write( '{:30}{:25}\n'.format( t[0], str(t[1]) ) )
