import nltk
import random
import numpy as np
from nltk.stem import WordNetLemmatizer

'''
+---------------------------------------------------------------+
|                      Logistic Regression                      |
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

def mixup( matrix , vectorY ):
    mapVectors = list( zip( matrix , vectorY ) )
    random.shuffle(mapVectors)
    mixMatrix , mixVector = zip(*mapVectors)
    mixMatrix = list(mixMatrix)
    mixVector = list(mixVector)
    lenTraining = int( len(mixMatrix) * 0.7)
    lenTest = len(mixMatrix) - lenTraining

    mTraining = mixMatrix[:lenTraining]
    vTraining = mixVector[:lenTraining]

    mTest = mixMatrix[-lenTest:]
    vTest = mixVector[-lenTest:]

    for row in mTraining:
        row.insert(0,1)
    for row in mTest:
        row.insert(0,1)
    return  mTraining , vTraining , mTest , vTest

def hypothesis( theta, matrix ):
    z = np.dot( theta , matrix )
    z = z.T
    h_x = 1 / ( 1 + np.exp(-z) )
    return h_x

def costFunction( h_x , y , m ):
    Y = y.reshape((m,1))
    y1 = np.multiply(Y, np.log(h_x))
    y0 = np.multiply(1-Y, np.log(1-h_x))
    cost = (-1 / m ) * np.sum(y1+y0)
    return cost

def gradientDescent( theta, h_x , y , alpha , matrix , m ):
    Y = y.reshape( (m,1) )
    d_theta = (1/m) * np.dot((h_x - Y).T, matrix)
    d_theta = d_theta.T
    theta = theta - alpha * d_theta
    return theta

if __name__ == '__main__':
    messages = readText( '../../Corpus/SMS_Spam_Corpus_big.txt' )
    matrix , vectorY = makeMatrix( messages )
    vocabulary , tokenMessage  = normalization( matrix )
    vectors = getVector( vocabulary , tokenMessage )
    matrix = getProbability( vectors )
    mTraining , vTraining , mTest , vTest = mixup( matrix , vectorY )

    test = np.array(mTest)
    matrix = np.array( mTraining )
    matrixT = matrix.T

    y = np.array(vTraining)
    yT = y.T
    auxY = np.array(vTest)
    vT = auxY.T

    theta = np.zeros( (len(mTraining[0]), 1) )
    thetaT = theta.T

    alpha = 0.3
    m = len(mTraining)

    print('''
    \t**************************************************
                            Training 
    \t**************************************************''')

    h_x = hypothesis( thetaT , matrixT )
    cost = costFunction( h_x , y , m  )   
    print('\tCost Function Initial: {}'.format(cost))

    for i in range(10000):
        auxTheta = gradientDescent( theta, h_x , y , alpha , matrix , m)
        theta = auxTheta
        thetaT = theta.T
        h_x = hypothesis( thetaT , matrixT )
        cost = costFunction( h_x , y , m  ) 
        if i % 50 == 0:
            print('\t{}.- Cost Function: {}'.format(i,cost))

    print('''
    \t**************************************************
                            Testing 
    \t**************************************************''')

    thetaT = theta.T
    for i,row in enumerate(test):
        h_x = float(hypothesis( thetaT, row ))
        p = 1 if h_x >= 0.5 else 0  
        print( '\tPrediction: {:22} => {:2}  Real: {:2}'.format( h_x ,p, vTest[i] ) )

    
