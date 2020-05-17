import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


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

def tfidf_extractor( corpus ):
    vectorizer = TfidfVectorizer(norm='l2',smooth_idf=True,use_idf=True)
    features = vectorizer.fit_transform( corpus )
    return vectorizer , features 

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

    return  mTraining , vTraining , mTest , vTest

if __name__ == '__main__':
    messages = readText( '../../Corpus/SMS_Spam_Corpus_big.txt' )
    matrix , vectorY = makeMatrix( messages )
    vectorizer,features = tfidf_extractor( matrix )
    matrix = np.round(features.todense(), 2)
    mTraining , vTraining , mTest , vTest = mixup( list(matrix) , vectorY )
    #Numpy vectors
    X_Training = np.array(mTraining)
    X_Test = np.array(mTest)
    Y_Training = np.array(vTraining)
    Y_Test = np.array(vTest)
    #Create Instance
    LR = LogisticRegression(random_state=0)
    #Training
    LR.fit( X_Training , Y_Training )
    #Predictions
    prediction = LR.predict(X_Test)
    for i,value in enumerate(prediction): 
        print( '\tPrediction: {:2}  Real: {:2}'.format( value , Y_Test[i] ) )
