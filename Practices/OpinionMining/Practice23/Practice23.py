import networkx
import numpy as np
from operator import itemgetter
from nltk.corpus import stopwords
from prettytable import PrettyTable
from scipy.sparse.linalg import svds
from nltk.stem import WordNetLemmatizer
from gensim.summarization import summarize
from nltk import word_tokenize, FreqDist, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

'''
+---------------------------------------------------------------+
|                           Summarization                       |
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
                #print( 'File: {} not found!' .format(file_name) , end='\r')
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
    sentences_text = [ ]
    for opinion in opinions_:
        for j,word in enumerate(words):
            if word in opinion:
                if j == 0:
                    sentences[0].append( opinion )
                elif j == 1:
                    sentences[1].append( opinion )
                elif j == 2:
                    sentences[2].append( opinion )
                elif j == 3:
                    sentences[3].append( opinion )
                elif j == 4:
                    sentences[4].append( opinion )
                elif j == 5:
                    sentences[5].append( opinion )
                else:
                    sentences[6].append( opinion )

    for sentence in sentences:
        aux = ''
        for sent in sentence:
            s = ''.join( sent )
            aux += s + '\n'
        sentences_text.append( aux )

    return sentences_text

def summary_gensim( words , sentences ):
    for i,word in enumerate(words):
        print('Resumen para: {}\n'.format(word) )
        if i == len(words)-1 :
            summary_ratio = 0.01
        else:
            summary_ratio = 0.03     
        print( summarize( sentences_yes[i] , ratio=summary_ratio , split=True ) )
        print('\n\t********************************************************************\n')

def summary_LSA( words , sentences ):
    norm_sentences = normalize_sentences( len(words) , sentences )
    sentences_sort = sentences_sorted( sentences )
    num_sentences = 2
    num_topics = 2

    for i, word in enumerate(words):
        print('Resumen para: {}\n'.format(word) )
        tv = TfidfVectorizer( min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform( norm_sentences[i] )
        dt_matrix = dt_matrix.toarray()
        td_matrix = dt_matrix.T
        u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)  
        term_topic_mat, singular_values, topic_document_mat = u, s, vt
        # remove singular values below threshold                                         
        sv_threshold = 0.5
        min_sigma_value = max( singular_values ) * sv_threshold
        singular_values[singular_values < min_sigma_value] = 0

        salience_scores = np.sqrt( np.dot( np.square(singular_values),np.square( topic_document_mat ) ) )
        top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
        top_sentence_indices.sort()
        print( '\n'.join( np.array( sentences_sort[i] )[top_sentence_indices] ) )
        print('\n\t********************************************************************\n')

    return

def normalize_sentences( num_words , sentences ):
    stop_words = stopwords.words('spanish')
    normalize_sentences = [ ]
    for i in range( num_words ):
        aux_sentences = sent_tokenize( sentences[i] )
        aux_sentence = [ ]
        for sentence in aux_sentences:
            tokens = word_tokenize( sentence )
            clean_tokens = [ token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha() ]
            aux = ''
            for tok in clean_tokens:
                union = ''.join(tok)
                aux += union + ' '
            aux_sentence.append( aux )
        normalize_sentences.append( aux_sentence )

    return normalize_sentences

def sentences_sorted( sentences ):
    sorted = [ ]
    for sentence in sentences:
        sorted.append( sent_tokenize( sentence , 'spanish' ) )
    
    return sorted

def low_rank_svd( matrix, singular_count=2 ):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def summary_text_rank( words , sentences ):
    norm_sentences = normalize_sentences( len(words) , sentences )
    sentences_sort = sentences_sorted( sentences )
    num_sentences = 2

    for i, word in enumerate(words):
        print('Resumen para: {}\n'.format(word) )
        tv = TfidfVectorizer( min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform( norm_sentences[i] )
        dt_matrix = dt_matrix.toarray()
        similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
        similarity_graph = networkx.from_numpy_array(similarity_matrix)
        scores = networkx.pagerank(similarity_graph)
        ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)
        top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
        top_sentence_indices.sort()
        try:
            print('\n'.join(np.array(sentences_sort[i])[top_sentence_indices]))
        except:
            print('No hay suficientes oraciones para el resumen')
        print('\n\t********************************************************************\n')

    return

if __name__ == '__main__':
    lemmas = openGenerate()
    op_no , tokens_no = read_texts( '../../Corpus/peliculas/no' , lemmas )
    op_yes , tokens_yes = read_texts( '../../Corpus/peliculas/yes' , lemmas)
    tokens_opinions = tokens_no + tokens_yes
    opinions_no = sent_tokenize( op_no , 'spanish' )
    opinions_yes = sent_tokenize( op_yes , 'spanish' )
    ngrams = get_top_ngrams( tokens_opinions , 1 , 30 )
    print( 'Unigrams:\n{}\n' .format(ngrams) )
    words = [ 'serie','personaje','historia','gustar','cine','escena','querer' ]
    sentences_yes = opinion_clasification( opinions_yes , words )
    sentences_no = opinion_clasification( opinions_no , words )
    print('\n\t************************ G E N S I M *******************************')
    print('\n\t******************* Opiniones positivas ****************************')
    summary_gensim( words , sentences_yes )
    print('\n\t******************* Opiniones negativas ****************************')
    summary_gensim( words , sentences_no )

    print('\n\t*************************** L S A ***********************************')
    print('\n\t******************* Opiniones positivas ****************************')
    summary_LSA( words , sentences_yes )
    print('\n\t******************* Opiniones negativas ****************************')
    summary_LSA( words , sentences_no )

    print('\n\t*************************** R A N K *********************************')
    print('\n\t******************* Opiniones positivas ****************************')
    summary_text_rank( words , sentences_yes )
    print('\n\t******************* Opiniones negativas ****************************')
    summary_text_rank( words , sentences_no )