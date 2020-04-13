import csv
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def readCSV():
    data = [ ]
    with open('Kc_house_data.csv', newline='') as File:  
        reader = csv.reader(File)
        for row in reader:
            data.append(list(row))
    data.pop(0)
    for row in data:
        row.pop(0)
    return data

def makeMatrix( data ):
    matrix = [ ]
    for i in range( len(data[0]) ):
        vector = [ ]
        for j in range( len(data) ):
            if i == 0:
                vector.append( int( data[j][i][-4:] ) )
            else:    
                vector.append( float( data[j][i] ) )
        matrix.append( vector )
    vectorY = matrix[1]
    matrix.pop(1)
    return matrix,vectorY

def normalize( matrix ):
    matrixN = [ ]
    for row in matrix:
        m = max( row ) - min(row)
        value = np.array( row )
        nu = np.average(value)
        vector = [ ]
        for r in row:
            vector.append( (r - nu)/m  )
        matrixN.append(vector)
    mapVectors = list( zip( matrixN[0] , matrixN[1] , matrixN[2] , matrixN[3],
                            matrixN[4] , matrixN[5] , matrixN[6] , matrixN[7],
                            matrixN[8] , matrixN[9] , matrixN[10], matrixN[11],
                            matrixN[12], matrixN[13], matrixN[14], matrixN[15],
                            matrixN[16], matrixN[17] ) )
    newMatrix = [ ]
    for m in mapVectors:
        newMatrix.append( list(m) )
    return newMatrix

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
    
def hyphotesis( matrix , theta ): 
    h_x = np.dot( theta,matrix )
    return h_x

if __name__ == '__main__':
    data = readCSV()
    matrix,vectorY = makeMatrix( data )
    matrix = normalize( matrix )
    mTraining , vTraining , mTest , vTest = mixup( matrix , vectorY )
    #Numpy vectors
    X_Training = np.array( mTraining )
    X_Test = np.array(mTest)
    Y_Training = np.array(vTraining)
    Y_Test = np.array(vTest)
    #Create instance 
    LR = LinearRegression ()
    #Traininig
    LR.fit( X_Training , Y_Training )
    #Predictions
    Y_Prediction = LR.predict( X_Test )
    for i, prediction in enumerate(Y_Prediction):
        print( '\tPrediction: {:13.5f}  Real: {:10}'.format( prediction , Y_Test[i] ) )
    #The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: {:.2f}'.format( r2_score(Y_Test, Y_Prediction) ) ) 
