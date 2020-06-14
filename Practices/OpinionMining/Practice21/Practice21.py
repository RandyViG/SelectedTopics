import numpy as np
import xml.dom.minidom
from operator import itemgetter
from nltk.corpus import stopwords
from prettytable import PrettyTable
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, FreqDist, sent_tokenize

'''
+---------------------------------------------------------------+
|                        Analisis de polaridad                  |
|                       apartir de diccionario                  |
+---------------------------------------------------------------+
'''
lenSet = 25

def openGenerate():
    f = open('../../Corpus/generate.txt',encoding='utf-8')
    lemmas = {}
    for line in f.readlines():
        lemmas[ line.split(' ')[0].replace('#','') ] = line.split(' ')[-1][:-1]
    f.close()
    return lemmas

def read_texts( path , lemmas ):
    opinions = ' '
    clear_opinions = ' '
    for j in range( 1 , 6 ):
        for i in range( 1 , lenSet ):
            tokens = [ ]
            file_name = path + '_' + str(j) + '_' + str(i) + '.txt'
            try:
                f = open( file_name , encoding = 'ISO-8859-1' )
                lines = f.readlines()
                tokens = [ word_tokenize(line) for line in lines if len( word_tokenize(line) ) != 0 ]
                clean_tokens = clean_opinions( tokens )
                tokens_lemmas = lemmatized( clean_tokens , lemmas )
                aux_tokens = lemmatized( tokens , lemmas )

                for line in tokens_lemmas:
                    opinion = ' '.join(line)
                    clear_opinions += opinion + ' '

                for line in aux_tokens:
                    opinion = ' '.join(line)
                    opinions += opinion + ' '


            except:
                print( 'File: {} not found!' .format(file_name) , end='\r')
                continue
    print('\n')
    tok_opinions = word_tokenize( clear_opinions )

    return opinions , tok_opinions

def clean_opinions( comments ):
    c_comments = [ ]
    stop_words = stopwords.words('spanish')
    for comment in comments:
        clean_comment = [ token.lower() for token in comment if token.lower() not in stop_words and token.isalpha() ]
        c_comments.append( clean_comment )
    
    return c_comments

def lemmatized( comments , lemmas):
    comm=[]
    for comment in comments:
        tokensLemmatized = []
        for tok in comment:
            try:
                tokensLemmatized.append(lemmas[tok])
            except KeyError:
                tokensLemmatized.append(tok)
        comm.append( tokensLemmatized )
    return comm

def get_top_ngrams( corpus , ngrams_val = 1 , limit = 5 ):

    corpus = flatten_corpus( corpus )
    tokens = word_tokenize( corpus )

    ngrams = compute_ngrams( tokens , ngrams_val )
    ngrams_freq_dist = FreqDist( ngrams )
    sorted_ngrams_fd = sorted( ngrams_freq_dist.items() , key =itemgetter(1) , reverse=True  )
    
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams =[ ( ' '.join(text) , freq ) for text, freq in sorted_ngrams ]

    return sorted_ngrams 

def compute_ngrams(sequence,n):
    return zip(*[sequence[index:] for index in range(n)])

def flatten_corpus( corpus ):
    return ' '.join( [ document.strip() for document in corpus] )

def read_polarities( path ):
    doc = xml.dom.minidom.parse( path )
    lemmas = doc.getElementsByTagName( 'lemma' )
    sentiCons = [ ]
    polarities = [ ]
    for lemma in lemmas:
        sentiCons.append( lemma.firstChild.data[1:-1] )
        polarities.append( float(lemma.attributes['pol'].value) )
    
    return sentiCons,polarities

if __name__ == '__main__':
    lemmas = openGenerate()
    opinions_no , tokens_no = read_texts( '../../Corpus/peliculas/no' , lemmas )
    opinions_yes , tokens_yes = read_texts( '../../Corpus/peliculas/yes' , lemmas)

    tokens_opinions = tokens_no + tokens_yes

    aux_opinions = opinions_no + opinions_yes
    opinions = sent_tokenize( aux_opinions , "spanish" )

    ngrams = get_top_ngrams( tokens_opinions , 1 , 20 )
    print( ngrams )
    
    words = [ 'película','personaje','historia','gustar','cine','creer','querer' ]

    sentences = [ [ ] , [ ] , [ ] , [ ] , [ ]  , [ ] , [ ] ]

    for i,opinion in enumerate(opinions):
        for j,word in enumerate(words):
            if word in opinion:
                if j == 0:
                    sentences[0].append( word_tokenize( opinion ) )
                elif j == 1:
                    sentences[1].append( word_tokenize( opinion ) )
                elif j == 2:
                    sentences[2].append( word_tokenize( opinion ) )
                elif j == 3:
                    sentences[3].append( word_tokenize( opinion ) )
                elif j == 4:
                    sentences[4].append( word_tokenize( opinion ) )
                elif j == 5:
                    sentences[5].append( word_tokenize( opinion ) )
                else:
                    sentences[6].append( word_tokenize( opinion ) )

    sentiCons , polarities = read_polarities('/home/randy/Descargas/ML-SentiCon/senticon.es.xml')
    
    sum_polarities = [ 0 , 0 , 0 , 0 , 0 , 0 ,0 ]
    num_polarities = [ 0 , 0 , 0 , 0 , 0 , 0 ,0 ]

    for i,s in enumerate(sentences):
        for sentence in s:
            for token in sentence:
                if token in sentiCons:
                    num_polarities[i] += 1
                    sum_polarities[i] += polarities[ sentiCons.index( token ) ]

    for i, word in enumerate(words):
        print( '{}.- {:10} {:20}'.format( i, word, ( sum_polarities[i] / num_polarities[i] ) ) )
