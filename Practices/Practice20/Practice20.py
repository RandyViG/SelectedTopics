import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from prettytable import PrettyTable
import xml.dom.minidom

'''
+---------------------------------------------------------------+
|                        Analisis de polaridad                  |
|                       apartir de diccionario                  |
+---------------------------------------------------------------+
'''
lenSet = 3000
def load_pkl(fname):
    from pickle import load
    f = open(fname, 'rb')
    loaded = load(f)
    f.close()
    return loaded

def save_pkl(a,fname):
    from pickle import dump
    output = open(fname, 'wb')
    dump(a, output, -1)
    output.close()

def read_texts( path ):
    comments = [ ]
    for i in range( 2 , lenSet ):
        tokens = [ ]
        file = path + str(i) + '.review.pos'
        try:
            f = open( file , encoding = 'ISO-8859-1' )
            lines = f.readlines()
            tokens = [word_tokenize(line) for line in lines]
            tokens_aux = [ ]
            for line in tokens:
                if len(line) > 0:
                    tokens_aux.append((line[1],line[2][0].lower()))
            comments.append(tokens_aux)
        except:
            print( 'File: {} not found!' .format(file) , end='\r')
            continue
            
    
    return comments

def clean_comments( comments ):
    c_comments = [ ]
    stop_words = stopwords.words('spanish')
    for comment in comments:
        clean_comment = [ token for token in comment if token[0] not in stop_words and token[0].isalpha() ]
        clean_comment=list(set(clean_comment))
        c_comments.append(clean_comment)
    
    return c_comments

def read_ranks( path ):
    ranks = [ ]
    for i in range( 2, lenSet ):
        file = path + str(i) + '.xml'
        try:
            f = open( file , encoding = 'ISO-8859-1' )
            line = f.readline()
            j = line.index( ' rank' )
            ranks.append( int( line[j+7] )  )
        except FileNotFoundError:
            print( 'File: {} not found!' .format(file) , end='\r' )

    return ranks

def read_polarities( path ):
    doc = xml.dom.minidom.parse( path )
    lemmas = doc.getElementsByTagName( 'lemma' )
    sentiCons = [ ]
    polarities = [ ]
    for lemma in lemmas:
        sentiCons.append( (lemma.firstChild.data[1:-1],lemma.attributes['pos'].value) )
        polarities.append( float(lemma.attributes['pol'].value) )
    return sentiCons,polarities

def get_polarities_comments( comments,polarities,sentiCons ):
    normalized_polarities = [ ]
    for i,comment in enumerate( comments):
        total = 0
        sum_p = 0
        print(i)
        for t in comment:
            if t in sentiCons:
                print(t)
                print(sentiCons[sentiCons.index(t)])
                total = total + 1
                sum_p = sum_p + polarities[ sentiCons.index(t) ]
        print(total)
        print(sum_p)
        try:
            normalized_polarities.append( sum_p/total )
        except ZeroDivisionError:
            normalized_polarities.append(0.0)
    return normalized_polarities

def get_polarities_class( normalized_polarities,ranks ):
    sumaC1=0.0
    C1P=0
    sumaC2=0.0
    C2P=0
    sumaC3=0.0
    C3P=0
    sumaC4=0.0
    C4P=0
    sumaC5=0.0
    C5P=0
    for i,p in enumerate( normalized_polarities ):
        if ranks[i]==1:
            sumaC1=sumaC1+p
            C1P=C1P+1
        elif ranks[i]==2:
            sumaC2=sumaC2+p
            C2P=C2P+1
        elif ranks[i]==3:
            sumaC3=sumaC3+p
            C3P=C3P+1
        elif ranks[i]==4:
            sumaC4=sumaC4+p
            C4P=C4P+1
        elif ranks[i]==5:
            sumaC5=sumaC5+p
            C5P=C5P+1
    p1=sumaC1/C1P
    p2=sumaC2/C2P
    p3=sumaC3/C3P
    p4=sumaC4/C4P
    p5=sumaC5/C5P

    return p1,p2,p3,p4,p5

if __name__ == "__main__":
    comments = read_texts( '/home/randy/Descargas/corpusCine/corpusCriticasCine/' )
    print('\n')
    comments = clean_comments( comments )
    save_pkl(comments,'comments.pkl')
    #comments=load_pkl('comments.pkl')
    ranks = read_ranks( '/home/randy/Descargas/corpusCine/corpusCriticasCine/' )
    save_pkl(ranks,'ranks.pkl')
    #ranks = load_pkl('ranks.pkl')
    print('\n')
    sentiCons,polarities = read_polarities('/home/randy/Descargas/ML-SentiCon/senticon.es.xml')
    
    save_pkl(sentiCons,'sentiCons.pkl')
    save_pkl(polarities,'polarities.pkl')
    #sentiCons = load_pkl('sentiCons.pkl')
    #polarities = load_pkl('polarities.pkl')
    print('\n')
    normalized_polarities = get_polarities_comments( comments,polarities,sentiCons )
    save_pkl(normalized_polarities,'normalized_polarities.pkl')
    #normalized_polarities = load_pkl('normalized_polarities.pkl')
    
    p1,p2,p3,p4,p5 = get_polarities_class( normalized_polarities,ranks )

    x = PrettyTable()
    x.field_names = ['Class','Polarities']

    x.add_row(['1',p1])
    x.add_row(['2',p2])
    x.add_row(['3',p3])
    x.add_row(['4',p4])
    x.add_row(['5',p5])
    
    print('''
    \t**************************************************
                      Analisis de polaridad 
    \t**************************************************''')
    print(x)
