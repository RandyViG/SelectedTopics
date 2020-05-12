import nltk
import random
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

def getCentroids( matrix ):
    random.seed()   
    c1 = randrange( 0 , len(matrix) )
    c2 = randrange( 0 , len(matrix) )
    return matrix[c1],matrix[c2]

def k_means( centroid1 , centroid2 , vectors , y ):

    for i in range( 50 ):
        #Cluster assigment
        cluster1 = [ ]
        cluster2 = [ ]
        y1 = [ ]
        y2 = [ ]
        for j,vector in enumerate(vectors):
            distance_C1 = sqrt ( np.sum( (vector - centroid1) ** 2 ) )
            distance_C2 = sqrt ( np.sum( (vector - centroid2) ** 2 ) )
            if distance_C1 < distance_C2:
                cluster1.append( vector )
                y1.append( y[j] )
            else:
                cluster2.append( vector )
                y2.append( y[j] )
        
        cluster1 = np.array( cluster1 )
        cluster2 = np.array( cluster2 )

        #move centroid
        aux = np.zeros( (len(cluster1[0]) ) )
        for vector in cluster1:
            aux = aux + vector
        centroid1 = aux * ( 1 / len( cluster1 ) )

        aux2 = np.zeros( (len(cluster2[0]) ) )
        for vector in cluster2:
            aux2 = aux2 + vector
        centroid2 = aux2 * ( 1 / len( cluster2 ) )

    return centroid1 , centroid2 , cluster1 , cluster2, y1 , y2

def costFunction( centroid1 , centroid2 , cluster1 , cluster2 , m):
    distance_C1 = 0
    distance_C2 = 0
    for vector in cluster1:
        distance_C1 = distance_C1 + sqrt ( np.sum( (vector - centroid1) ** 2 ) )
    for vector in cluster2:
        distance_C2 = distance_C2 + sqrt ( np.sum( (vector - centroid2) ** 2 ) )

    return ( ( distance_C1 + distance_C2 ) / m )

if __name__ == '__main__':
    messages = readText( 'spam.txt' )
    vectors , vectorY = makeMatrix( messages )
    vocabulary , tokenMessage  = normalization( vectors )
    vectors = getVector( vocabulary , tokenMessage )
    proba_vectors = getProbability( vectors )
    vectors = np.array( proba_vectors )
    m = len( vectors )
    costs = [ ]
    values_y1 = [ ]
    values_y2 = [ ]
    for i in range( 50 ):
        print('Iteration {}'.format(i) , end="\r")
        centroid1,centroid2 = getCentroids( vectors )
        centroid1,centroid2,cluster1,cluster1,y1,y2 = k_means( centroid1 , centroid2 , vectors , vectorY )
        cost = costFunction( centroid1 , centroid2 , cluster1 , cluster1 , m )
        costs.append( ( cost,centroid1,centroid2 ) )
        values_y1.append( y1 )
        values_y2.append( y2 )
    print("\n\r",end="")
    order = sorted( costs , key=lambda tup:tup[0] )
    i = costs.index( order[0] )

    vector_y1 = values_y1[i]
    vector_y2 = values_y2[i]
    print(len(vector_y1))
    print(len(vector_y2))

    print('Cost: {}'.format(order[0][0]))
    print('Spam Cluster 1 {}'.format( vector_y1.count(1) ) )
    print('Ham Cluster 1 {}'.format( vector_y1.count(0) ) )
    print('Spam Cluster 2 {}'.format( vector_y2.count(1) ) )
    print('Ham Cluster 2 {}'.format( vector_y2.count(0) ) )


