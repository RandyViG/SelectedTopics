import nltk
from random import randrange, seed
import numpy as np
from math import sqrt
from random import randrange
from prettytable import PrettyTable
from nltk.stem import WordNetLemmatizer

'''
+---------------------------------------------------------------+
|                               K - means                       |
+---------------------------------------------------------------+
'''

def readText( nameFile ):
    messages = [ ]
    with open( nameFile, encoding='iso-8859-1') as f:
        for line in f:
            messages.append(line.replace('\n',''))
    f.close()
    return messages

def makeMatrix( messages ):
    matrix = [ ]
    vector = [ ]
    for message in messages:
        if message[-4:] == 'spam':
            vector.append(1)
            matrix.append( message[:-5] )
        else:
            vector.append(0)
            matrix.append( message[:-4] )
    return matrix,vector

def normalization( messages ):
    token = [ ]
    vocabulary = [ ]
    for message in messages:
        t = tokens( message )
        token.append(t)
        vocabulary = vocabulary + t
    vocabulary = sorted(set(vocabulary))
    return vocabulary,token

def tokens( message ):
    words = nltk.word_tokenize( message )
    words = [ w.lower() for w in words ]
    words = nltk.pos_tag(words)
    words = lemmas( words )
    words = cleanTagger( words ) 
    return words

def lemmas( tokens ):
    lem = [ ]
    wnl = WordNetLemmatizer()
    for token in tokens:
        l = wnl.lemmatize( token[0] )
        lem.append( ( l , token[1] ) )
    return lem

def cleanTagger(s_tagged):
    tags = [ ]
    for i in range(len(s_tagged)):
        tags.append(s_tagged[i][0]+" "+s_tagged[i][1])
    return tags

def getVector( vocabulary , messages ):
    vectors = [ ]
    for message in messages:
        vector = [ ]
        for word in vocabulary:
            frec = message.count(word)
            vector.append(frec)
        vectors.append( vector )
    return vectors

def getProbability( vectors ):
    probabilities = [ ]
    for vector in vectors:
        proba = [ ]
        value = np.array( vector )
        suma = np.sum( value )
        for v in vector:
            proba.append(v/suma)
        probabilities.append(proba)
    return probabilities

def getCentroids(vectors):
    seed( None )
    centroid1 = randrange( len(vectors) )
    centroid2 = randrange( len(vectors) )
    centroid1 = vectors[ centroid1 ]
    centroid2 = vectors[ centroid2 ]

    return centroid1,centroid2

def kmeans( centroid1 , centroid2 , vectors ,y):
    for i in range(50):
        #Cluster assignment
        cluster1=[]
        cluster2=[]
        y1=[]
        y2=[]
        for j,vector in enumerate(vectors):
            distance1 = sqrt(np.sum( (vector-centroid1) ** 2) )
            distance2 = sqrt(np.sum( (vector-centroid2) ** 2) )
            if distance1 < distance2:
                cluster1.append(vector)
                y1.append(y[j])
            else:
                cluster2.append(vector)
                y2.append(y[j])
        
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
        
        #Move centroid
        aux = np.zeros( ( len( cluster1[0]) ) )
        for vector in cluster1:
            aux = aux + vector
        centroid1 = aux * (1/len(cluster1))
        aux2 = np.zeros( ( len( cluster2[0]) ) )
        for vector in cluster2:
            aux2 = aux2 + vector
        centroid2 = aux2 * ( 1/len(cluster2) )

    return centroid1,centroid2,cluster1,cluster2,y1,y2


def costFunction(centroid1,centroid2,cluster1,cluster2,m):
    suma1=0
    suma2=0
    for vector in cluster1:
        distance1 = sqrt ( np.sum( (vector-centroid1) ** 2 ) )
        suma1=suma1+distance1
    for vector in cluster2:
        distance2 = sqrt ( np.sum( (vector-centroid2) **2 ) )
        suma2=suma2+distance2
    suma=suma1+suma2
    cost=suma/m
    
    return cost
    
if __name__ == '__main__':
    messages = readText('../../Corpus/SMS_Spam_Corpus_big.txt')
    vectors,vector_y = makeMatrix( messages )
    vocabulary,tokens = normalization( vectors )
    vectors = getVector( vocabulary , tokens )
    probabilities_vectors = getProbability( vectors )
    vectors = np.array( probabilities_vectors )

    m = len( vectors )
    costs = [ ]
    tables = [ ]
    for i in range(50):
        table = PrettyTable()
        table.field_names = ['  ', 'Numero de spam' , 'Numero de ham']
        centroid1,centroid2 = getCentroids(vectors)
        centroid1,centroid2,cluster1,cluster2,y1,y2 = kmeans( centroid1 , centroid2 , vectors , vector_y )
        cost = costFunction( centroid1 , centroid2 , cluster1 , cluster2 , m)
        costs.append( cost )
        table.add_row(['Cluster1',y1.count(1),y1.count(0)])
        table.add_row(['Cluster2',y2.count(1),y2.count(0)])
        tables.append(table)

    min_cost = min( costs )
    i = costs.index( min_cost )
    print( 'Costo Minimo: {} '.format(min_cost) )
    print(tables[i])