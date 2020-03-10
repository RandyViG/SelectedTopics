import nltk
import re
import math 
import operator
from pickle import dump, load
from bs4 import BeautifulSoup
from nltk.corpus import cess_esp
from nltk.corpus import stopwords
from prettytable import PrettyTable
from nltk.tokenize import RegexpTokenizer

def openText():
    f = open('../Corpus/e961024.htm', encoding = 'utf-8')
    text = f.read()
    f.close()

    return text

def getArticles( text ):
    articles = [ ]
    segments = re.split('<h3>',text)
    for s in segments:
        soup = BeautifulSoup(s,'lxml')
        t = soup.get_text()
        articles.append(t)

    return articles


def openGenerate():
    f = open('../generate.txt',encoding='utf-8')
    lemmas = {}
    for line in f.readlines():
        lemmas[ line.split(' ')[0].replace('#','') ] = line.split(' ')[-1][:-1]
    f.close()
    
    return lemmas

def normalization( articles ):
    stop = stopwords.words('spanish')
    articlesTokens = [ ]
    for art in articles:
        t = tokens(art)
        t = [w for w in t if w.lower() not in stop]
        articlesTokens.append(t)

    return articlesTokens

def lemmatized( articlesTokens , lemmas):
    articlesLemmatized = [ ]
    for art in articlesTokens:
        tokensLemmatized = [ ]
        for tok in art:
            try:
                tokensLemmatized.append(lemmas[tok])
            except KeyError:
                tokensLemmatized.append(tok)
        articlesLemmatized.append( tokensLemmatized )

    return articlesLemmatized 

def cleaningText(text):
    soup = BeautifulSoup(text, 'lxml') 
    text = soup.get_text()
    
    return text

def tokens(text):
    words = nltk.word_tokenize(text)
    words=[w.lower() for w in words if w.isalpha()]
    
    return words

def getPickle(fileName): 
    with open(fileName,'rb') as f:
        return load(f)


if __name__ == '__main__':

    table = PrettyTable()
    #topics = ['crisis ncfn000','privatización ncfs000','contaminación ncfs000','política ncfs000','economía ncfs000','tecnología ncfs000','Televisa']  
    topics = ['crisis','privatización','contaminación','política','economía','tecnología','Televisa']
    text = openText()
    articles = getArticles( text )
    lemmas = openGenerate()
    articlesTokens = normalization( articles )
    tokenLemma = lemmatized( articlesTokens , lemmas )
    topicMining = [ ]
    for t in topics:
        appear = [ ]
        for token in tokenLemma:
            appear.append( ( token.count(t) / 7 ) * 100 )
        topicMining.append( appear )
    header = [ 'Topic' ]
    rows = [ ]
    for i,art in enumerate( articles ):
        header.append( 'Article '+ str(i) )
    table.field_names = header
    for i,t in enumerate( topics ):
        r = [t] + topicMining[i]
        rows.append( r )
    for r in rows:
        table.add_row(r)
    print(table)      
    