# Requires the latest pip
#pip install --upgrade pip

# Current stable release for CPU and GPU
#pip install tensorflow

from tensorflow import keras

import PUJ
import numpy 
import pandas as pd

#dataset = keras.datasets.mnist.load_data(path="mnist.npz")
#print(dataset)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_test_dummy = pd.get_dummies(y_test).to_numpy()
y_train_dummy = pd.get_dummies(y_train).to_numpy()

x_train_reshape = x_train.reshape(60000, 28*28)
x_test_reshape = x_test.reshape(10000, 28*28)

y_train_dummy_2 = y_train_dummy[:,0].reshape(60000, 1)

data = numpy.concatenate((x_train_reshape, y_train_dummy_2), axis=1)

m = PUJ.Model.Logistic( )
m.setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
print( 'Initial model = ' + str( m ) )
 

# Configure cost
J = PUJ.Model.Logistic.Cost( m, data[ : , 0 : -1 ], data[ : , -1 : ] )

# Debugger
debugger = PUJ.Optimizer.Debug.Simple
##debugger = PUJ.Optimizer.Debug.PlotPolynomialCost( data[ : , 0 : -1 ], data[ : , -1 : ] )

# Fit using an optimization algorithm
opt = PUJ.Optimizer.GradientDescent( J )
opt.setDebugFunction( debugger )
opt.setLearningRate( 1e-5 )
opt.setNumberOfIterations( 10000 )
opt.setNumberOfDebugIterations( 1000 )
opt.Fit( )

# Show results
print( '===========================================' )
print( '= Iterations       :', opt.realIterations( ) )
print( '= Fitted model     :', m )
print( '===========================================' )

y_real = data[ : , -1 : ]
y_est = m.threshold( data[ : , 0 : -1 ] )
K = numpy.zeros( ( 2, 2 ) )

for i in range( y_real.shape[ 0 ] ):
  K[ int( y_real[ i, 0 ] ), int( y_est[ i, 0 ] ) ] += 1
# end for

print( K )
