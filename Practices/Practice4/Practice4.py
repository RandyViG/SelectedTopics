import nltk
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import operator
from pickle import dump, load
from nltk.corpus import cess_esp


def openText():
    f = open('../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()
    return text

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    return text

def tokens(text):
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    return words


def getVector(vocabulary,contexts):
    vectors={}
    for termino in vocabulary:
        context=contexts[termino]
        vector=[]
        for word in vocabulary:
            w=word.split(' ')
            frec=context.count(w[0])
            vector.append(frec)
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

#new
def normalization(text):
    stop=stopwords.words('spanish')
    t=cleaningText(text)
    t=tokens(t)
    t=[w for w in t if w.lower() not in stop]
    vocabulary=sorted(set(t))
    return vocabulary,t

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
    input=open(fname, 'rb')
    default_tagger=load(input)
    input.close()  
    s_tagged=combined_tagger.tag(clean_vocabulary)
    #s_tagged=default_tagger.tag(clean_vocabulary)
    return s_tagged

def cleanTagger(s_tagged):
    list(set(s_tagged))
    vocabulary=[]
    for i in range(len(s_tagged)):
        vocabulary.append(s_tagged[i][0]+" "+s_tagged[i][1])
    return vocabulary

def getContext(vocabulary,cleanWords):
    contexts = {}
    for termino in vocabulary:
        context = []
        for j,word in enumerate(cleanWords):
            term=termino.split(' ')
            #print(term[0])
            if term[0] == word :
                context+=cleanWords[ 0 if j-4 < 0 else j-4 : j ] 
                context+= cleanWords[ j+1 : j+5 if j < len(cleanWords) else len(cleanWords) ] 
        contexts[termino] = context
    return contexts

def stemming(tokens):
    newTokens=[]
    fopen="../generate.txt"
    archivo= open(fopen,encoding='utf-8')
    lemmas={}
    for linea in archivo.readlines(): 
        lemmas[linea.split(' ')[0].replace("#","")]=linea.split(" ")[0][:linea.split(" ")[0].find("#")]
    archivo.close()
    #for key in lemmas:
    #    print (key, ":", lemmas[key])
    
    for token in tokens:
        if token in lemmas:
            newTokens.append(lemmas[token])
        else:
            newTokens.append(token)
    return newTokens

def backTokens(vectorCosines,tokens,tokensTerm):
    back={}
    for j,key in enumerate(vectorCosines):
        if key in tokensTerm:
            vectorCosines[tokens[j]]=vectorCosines.pop(key)
    back=vectorCosines

    return back

if __name__ == "__main__":
      
    text=openText()
    vocabulary,tokens=normalization(text)
    tokensTerm=stemming(tokens)
    
    s_tagged=generateTagger(tokensTerm)
    print("entrenando")
    final_vocabulary=cleanTagger(s_tagged)
    print("get context")
    #print(final_vocabulary)
    context=getContext(final_vocabulary,tokensTerm)
    print("get vector")
    vectors=getVector(final_vocabulary,context)
    word="grande aq0cs0"
    print("get cosines")
    vectorCosines=getCosines(final_vocabulary,vectors,word)
    print("get ordenando")
    vectorCosines=sorted(vectorCosines.items(),key=lambda kv: kv[1],reverse=True)
    #vectorCosines=backTokens(vectorCosines,vocabulary,tokensTerm)
    print("imprimir")
    fv=open("similitudes.txt","w")
    #for item in vectorCosines:
    #    fv.write("%s\n" %(item,))
    for key in vectorCosines:
        if (key[0].split(" ")[1][0] == 'a'):
            fv.write(key[0].split(" ")[0]+" "+key[0].split(" ")[1][0]+" "+str(key[1])+'\n')
    fv.close()
    