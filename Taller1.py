# Requires the latest pip
#pip install --upgrade pip

# Current stable release for CPU and GPU
#pip install tensorflow

from tensorflow import keras

import PUJ
import numpy 
import pandas as pd

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train_dummy = pd.get_dummies(y_train).to_numpy()
 
x_train_reshape = x_train.reshape(60000, 28*28)
x_test_reshape = x_test.reshape(10000, 28*28)

#### Fijar semilla y reducir lineas 
mod = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4","Modelo_5","Modelo_6","Modelo_7","Modelo_8","Modelo_9"] 

for modelos in range(10):
    data = numpy.concatenate((x_train_reshape, y_train_dummy[:,modelos].reshape(60000, 1)), axis=1)

    mod[modelos] = PUJ.Model.Logistic( )
    mod[modelos].setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
    #print( 'Initial model = ' + str( mod ) )
    
    # Configure cost
    J = PUJ.Model.Logistic.Cost( mod[modelos], data[ : , 0 : -1 ], data[ : , -1 : ] )
    
    # Debugger
    debugger = PUJ.Optimizer.Debug.Simple
    
    # Fit using an optimization algorithm
    opt = PUJ.Optimizer.GradientDescent( J )
    opt.setDebugFunction( debugger )
    opt.setLearningRate( 1e-7 )
    opt.setNumberOfIterations( 500 )
    opt.setNumberOfDebugIterations( 100 )
    opt.Fit( )

def Calificacion(data):   
    y_prob = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4","Modelo_5","Modelo_6","Modelo_7","Modelo_8","Modelo_9"]
    for modelos in range(10):
        y_prob[modelos] = mod[modelos].evaluate(data)
    clients = pd.concat([pd.DataFrame(y_prob[0]), pd.DataFrame(y_prob[1]), pd.DataFrame(y_prob[2]), pd.DataFrame(y_prob[3]), pd.DataFrame(y_prob[4]), pd.DataFrame(y_prob[5]), pd.DataFrame(y_prob[6]), pd.DataFrame(y_prob[7]), pd.DataFrame(y_prob[8]), pd.DataFrame(y_prob[9])], axis=1,)
    clients.columns =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    Valor = clients.idxmax(axis = 1) 
    return Valor
#%%
### Casos favorables sobre casos totales, la diagonal muestra que estan bien clasificados TRAIN
y_real = y_train.reshape(60000, 1)
y_est = Calificacion(x_train_reshape)
K = numpy.zeros( ( 10, 10 ) )

for i in range( y_real.shape[ 0 ] ):
  K[ int( y_real[ i ] ), int( y_est[ i ] ) ] += 1
# end for
Aciertos = numpy.trace(K)/y_real.shape[ 0 ]
print(Aciertos)

#%%
### Casos favorables sobre casos totales, la diagonal muestra que estan bien clasificados TEST
y_real_test = y_test.reshape(10000, 1)
y_est_test = Calificacion(x_test_reshape)
K = numpy.zeros( ( 10, 10 ) )

for i in range( y_est_test.shape[ 0 ] ):
  K[ int( y_real_test[ i ] ), int( y_est_test[ i ] ) ] += 1
# end for
Aciertos = numpy.trace(K)/y_real_test.shape[ 0 ]
print(Aciertos)
