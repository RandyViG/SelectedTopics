import numpy as np
from operator import itemgetter
from nltk.corpus import stopwords
from prettytable import PrettyTable
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, FreqDist, sent_tokenize

'''
+---------------------------------------------------------------+
|                  Analisis de polaridad a detalle              |
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

def opinion_clasification( opinions_, words ):
    sentences = [ [ ] , [ ] , [ ] , [ ] , [ ]  , [ ] , [ ] ]

    for opinion in opinions_:
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

    return sentences

def read_lexicon( path ):
    lexicon = { }
    files = [ 'fullStrengthLexicon.txt' , 'mediumStrengthLexicon.txt' ]
    for file in files:
        with open( path + '/' + file , 'r') as f:
            lines = f.readlines()
            for line in lines:
                index = line.find('\t')
                if line[-1:] == '\n':
                    lexicon[ line[ : index ] ] = line[ -4 : -1 ]
                else:
                    lexicon[ line[ : index ] ] = line[ -3 : ]
        f.close()

    return lexicon

def get_words_polarities( sentences_ , lexicon):
    positive = [ 0 , 0 , 0 , 0 , 0 , 0 , 0  ]
    negative = [ 0 , 0 , 0 , 0 , 0 , 0 , 0  ]
    yes = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    aux_words_positive = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    aux_words_negative = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    words_positive = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    words_negative = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]

    for i,sentences in enumerate( sentences_ ):
        for sentence in sentences:
            yes[ i ] += sentence
            for w in sentence:
                try:
                    op = lexicon[w]
                    if op == 'pos':
                        aux_words_positive[i].append( w )
                        positive[i] += 1
                    else:
                        aux_words_negative[i].append( w )
                        negative[i] += 1
                except KeyError:
                    pass
    
    for i,words in enumerate(aux_words_positive):
        words_positive[i] = set( words )
    
    for i,words in enumerate(aux_words_negative):
        words_negative[i] = set( words )

    return yes , words_positive, words_negative, positive, negative
    
def get_probabilities( sentences_ , words_positive , words_negative ):
    probabilities = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
    words_probabilities_positive = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    words_probabilities_negative = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    sorted_positive = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]
    sorted_negative = [ [ ] , [ ] , [ ] , [ ] , [ ] , [ ] , [ ] ]

    for i,sentence in enumerate(sentences_):
        set_sentence = set( sentence )
        for word in set_sentence:
            if word.isalpha():
                frec = sentence.count( word )
                probabilities[i] += frec
                
    for i,words in enumerate(words_positive):
        for word in words:
            frec = yes[i].count( word )
            words_probabilities_positive[i].append( ( frec/probabilities[i] , word ) )
    
    for i,words in enumerate(words_negative):
        for word in words:
            frec = yes[i].count( word )
            words_probabilities_negative[i].append( ( frec/probabilities[i] , word ) )

    for i,words in enumerate( words_probabilities_positive ) :
        sorted_positive[i] = sorted( words )
    
    for i,words in enumerate( words_probabilities_negative ) :
        sorted_negative[i] = sorted( words )
    
    return sorted_negative , sorted_positive


if __name__ == '__main__':
    lemmas = openGenerate()
    op_no , tokens_no = read_texts( '../../Corpus/peliculas/no' , lemmas )
    op_yes , tokens_yes = read_texts( '../../Corpus/peliculas/yes' , lemmas)
    tokens_opinions = tokens_no + tokens_yes
    opinions_no = sent_tokenize( op_no , 'spanish' )
    opinions_yes = sent_tokenize( op_yes , 'spanish' )
    ngrams = get_top_ngrams( tokens_opinions , 1 , 20 )
    print( 'Unigrams:\n{}\n' .format(ngrams) )
    words = [ 'película','personaje','historia','gustar','cine','creer','querer' ]
    sentences_yes = opinion_clasification( opinions_yes , words )
    sentences_no = opinion_clasification( opinions_no , words )
    lexicon = read_lexicon( '../../Corpus/Spanish_sentiment_lexicon/' )
    yes,w_positive_yes,w_negative_yes,positive_yes,negative_yes = get_words_polarities( sentences_yes , lexicon )
    no,w_positive_no,w_negative_no,positive_no,negative_no = get_words_polarities( sentences_no , lexicon )
    w_sort_pos_yes,w_sort_neg_yes = get_probabilities( yes,w_positive_yes,w_negative_yes ) 
    w_sort_pos_no,w_sort_neg_no = get_probabilities( no,w_positive_no,w_negative_no ) 

    print( '{:^20} {:30} {:30}\n'.format('Caracteristicas','Polaridad en opiniones yes','Polaridad en opiniones no'))
    for i,word in enumerate( words ):
        value1 = ''
        if positive_yes[i] > negative_yes[i]:
            value1 = 'pos'
        elif positive_yes[i] < negative_yes[i]:
            value1 = 'neg'
        else:
            value1 = 'neutro'
        value2 = ''
        if positive_no[i] > negative_no[i]:
            value2 = 'pos'
        elif positive_no[i] < negative_no[i]:
            value2 = 'neg'
        else:
            value2 = 'neutro'
        print( '{:^20} {:^30} {:^30}'.format(word,value1,value2) )
    
    print('\n')
    print( '{:^20} {:65}      {:65}\n'
            .format('Caracteristicas',
            '5 Palabras positivas con la probabilidad mas alta en opniones yes',
            '5 Palabras negativas con la probabilidad mas alta en opniones yes') )
    for i,word in enumerate( words ):
        for j in range(1,6):
            pos1 = len(w_sort_pos_yes[i])-j
            pos2 = len(w_sort_neg_yes[i])-j
            if j == 3:
                print( '{:^20} {:^30} {:^35}      {:^30} {:^35}'.format(word,w_sort_pos_yes[i][pos1][1],w_sort_pos_yes[i][pos1][0],w_sort_neg_yes[i][pos2][1], w_sort_neg_yes[i][pos2][0] ) )
            else:
                print( '{:^20} {:^30} {:^35}      {:^30} {:^35}'.format('',w_sort_pos_yes[i][pos1][1],w_sort_pos_yes[i][pos1][0],w_sort_neg_yes[i][pos2][1], w_sort_neg_yes[i][pos2][0] ) )
        print('\n')


    print('\n')
    print( '{:^20} {:65}      {:65}\n'
            .format('Caracteristicas',
            '5 Palabras positivas con la probabilidad mas alta en opniones no',
            '5 Palabras negativas con la probabilidad mas alta en opniones no') )
    for i,word in enumerate( words ):
        for j in range(1,6):
            pos1 = len(w_sort_pos_no[i])-j
            pos2 = len(w_sort_neg_no[i])-j
            if j == 3:
                print( '{:^20} {:^30} {:^35}      {:^30} {:^35}'.format(word,w_sort_pos_no[i][pos1][1],w_sort_pos_no[i][pos1][0],w_sort_neg_no[i][pos2][1], w_sort_neg_no[i][pos2][0] ) )
            else:
                print( '{:^20} {:^30} {:^35}      {:^30} {:^35}'.format('',w_sort_pos_no[i][pos1][1],w_sort_pos_no[i][pos1][0],w_sort_neg_no[i][pos2][1], w_sort_neg_no[i][pos2][0] ) )
        print('\n')

