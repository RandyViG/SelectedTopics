import csv
import random
import numpy as np
import matplotlib.pyplot as plt

'''
+---------------------------------------------------------------+
|                      Linear Regression                        |
|                           Analysis                            |
+---------------------------------------------------------------+
'''

def readCSV():
    data = [ ]
    with open('../../Corpus/Kc_house_data.csv', newline='') as File:  
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
            vector.append( (r - nu)/m ) 
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
    mixMatrix = list( mixMatrix )
    mixVector = list( mixVector )
    lenTraining = int( len(mixMatrix) * 0.6 )
    lenTest = int( ( len(mixMatrix) - lenTraining ) / 2 )
    indexTest = lenTest + lenTraining

    mTraining = mixMatrix[ : lenTraining ]
    vTraining = mixVector[ : lenTraining ]

    mTest = mixMatrix[ lenTraining : indexTest ]
    vTest = mixVector[ lenTraining : indexTest ]

    mVC = mixMatrix[ indexTest : ]
    vVC = mixVector[ indexTest : ]

    for row in mVC:
        row.insert(0,1)

    for row in mTraining:
        row.insert(0,1)
    
    for row in mTest:
        row.insert(0,1)

    return  mTraining , vTraining , mTest , vTest , mVC , vVC
    
def hyphotesis( matrix , theta ): 
    h_x = np.dot( theta,matrix )
    return h_x

def costFunction( m , y , h_x , theta , Lambda ):
    valCost = np.subtract( h_x , y )
    powCost = np.power( valCost , 2 )
    sumCost = float( np.sum( powCost ) ) + Lambda * float( np.sum( np.power(theta,2) ) )
    return sumCost * (1 / (2*m) )

def costF( m , y , h_x ):
    valCost = np.subtract( h_x , y )
    powCost = np.power( valCost , 2 )
    sumCost = float( np.sum( powCost ) )  * (1 / (2*m) )
    return sumCost

def gradientDescent( theta, h_x , y , alpha , matrixT , m ):
    auxTheta = [ ]
    for j in range( len(matrixT) ):
        summatory = 0
        for i in range( len(matrixT[j]) ):
            res = ((h_x[0][i] - y[i])) * matrixT[j][i]
            summatory = summatory + res
        summatory = summatory / m
        aux =  theta[j][0] - (alpha * summatory)
        listAux = []
        listAux.append(aux)
        auxTheta.append(listAux)
    newTheta = np.array(auxTheta)
    return newTheta


if __name__ == '__main__':
    data = readCSV()
    matrix,vectorY = makeMatrix( data )
    matrix = normalize( matrix )
    mTraining , vTraining , mTest , vTest , mVC , vCV = mixup( matrix , vectorY )

    test = np.array(mTest)
    matrix = np.array( mTraining )
    matrixT = matrix.T
    cv = np.array( mVC )

    y = np.array(vTraining)
    yT = y.T
    auxY = np.array(vTest)
    vT = auxY.T
    vC = np.array( vCV )


    theta = np.zeros( (len(mTraining[0]), 1) )
    thetaT = theta.T

    alpha = 0.1
    m = len(mTraining)
    print('''
    \t**************************************************
                            Training 
    \t**************************************************''')

    '''
    cost_CV = [ ]
    la = [ ]

    Lambda = 0.0
    name = 'Lambda_'
    for l in range(10):
        print( 'Iteracion: {}' .format(l ) )

        h_x = hyphotesis( matrixT , thetaT )
        cost = costFunction( m , yT , h_x , thetaT , Lambda )

        for i in range( 100 ):
            auxTheta = gradientDescent( theta, h_x , yT , alpha , matrixT , m)
            theta = auxTheta
            thetaT = theta.T
            h_x = hyphotesis( matrixT , thetaT )
            cost = costFunction( m , yT , h_x , thetaT , Lambda)

        h_x = hyphotesis( cv.T , thetaT  )
        cost = costF( m, vC , h_x )
        cost_CV.append( cost )
        la.append( Lambda )
        print('Cost {}'.format(cost))

        #plt.plot( list(range( 50 )) , h_x.T[ : 50 ] , 'r-o' )
        #plt.plot( list(range( 50 )) , vC[ : 50 ] , 'c-o' )
        #plt.title(  'Lambda: {:.4f}'.format( Lambda)  )
        #plt.savefig( 'Error_{:.4f}.png'.format( Lambda) )
        #plt.close() 
        
        Lambda = Lambda + 0.02
    print( cost_CV )
    #plt.plot( la , cost_train , 'c-3' )
    plt.plot( la , cost_CV , 'r-o' )
    plt.title( 'Error' )
    plt.xlabel( 'Lambda' )
    plt.ylabel( 'Error' )
    plt.savefig('Error.png')
    #plt.show()

    print( 'Error Min {}'.format( min(cost_CV) ) )
    print( 'Lambda: {} ' .format( la[ cost_CV.index(min(cost_CV)) ] ) )

    '''
    Lambda = 0.18

    h_x = hyphotesis( matrixT , thetaT )
    cost = costFunction( m , yT , h_x , thetaT , Lambda )

    for i in range( 1000 ):
        print(i)
        auxTheta = gradientDescent( theta, h_x , yT , alpha , matrixT , m)
        theta = auxTheta
        thetaT = theta.T
        h_x = hyphotesis( matrixT , thetaT )
        cost = costFunction( m , yT , h_x , thetaT , Lambda)

    print('''
    \t**************************************************
                            Testing 
    \t**************************************************''')

    thetaT = theta.T

    for i,row in enumerate(test):
        h_x = float(hyphotesis( row , thetaT ))
        print( '\tPrediction: {:20}  Real: {:10}'.format( h_x , vT[i] ) )

    h_x = hyphotesis( test.T , thetaT )
    plt.plot( list(range(50)) , h_x[:50] , 'r-o' )
    plt.plot( list(range(50)) , vT[:50] , 'c-o' )
    plt.title( 'test' )
    plt.savefig('test.png')

