# ----------------------------------------------------------------------------------------
# PROGRAMA: <<Taller 1: identicar dígitos>>
# ----------------------------------------------------------------------------------------
# Descripción: <<Este es un programa que aprende como reconocer números en imagenes por medio de 10 modelos logisticos. La clasificación se elige 
# a partir de la categoria con probabilidad mas alta. Posteriormente se crea una función para poder identificar digitos con el modelo ya entrenado>>
# ----------------------------------------------------------------------------------------
# Autores:
''' 
# Miguel David Benavides Galindo md_benavidesg@javeriana.edu.co
# Marlon Jaramillo Zapata        marlon.jaramillo@javeriana.edu.co
# Katherine Xiomar González      gonzalezskatherinex@javeriana.edu.co
'''
# Version: 1.0
# [01.03.2022]
# ----------------------------------------------------------------------------------------
# IMPORTAR MODULOS

# Requires the latest pip
#pip install --upgrade pip

# Requires tensorflow
#pip install tensorflow

from tensorflow import keras
import PUJ
import numpy 
import pandas as pd
import cv2 

# ----------------------------------------------------------------------------------------
# Solución taller 
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# 1) Recuperar la base de datos MNIST
# ----------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# ----------------------------------------------------------------------------------------
# 2) Proponer un sistema, que use únicamente la regresión logística, para identicar los dígitos del 0 al 9:
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# 2.1) Diseñar el proceso de entrenamiento.
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Nombre: <<Aprendizaje de digitos en imagenes>>
# ----------------------------------------------------------------------------------------
# Descripcion: <<Con la base Mnist, se toma las etiquetas y se convierten a variables Dummy, luego las imagenes 
# se transforman a dos dimensiones, ya con los datos en dos dimensiones se entrenan diez modelos logisticos usando,
# la librería del profe uno por cada categoria Dummy, la decisión se toma a partir de la categoria con probabilidad 
# mas alta de todos loos modelos>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
# La función necesita la base Mnist
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna los 10 modelos ya entrenados 
# ----------------------------------------------------------------------------------------
y_train_dummy = pd.get_dummies(y_train).to_numpy()
x_train_reshape = x_train.reshape(60000, 28*28)
x_test_reshape = x_test.reshape(10000, 28*28)

mod = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4","Modelo_5","Modelo_6","Modelo_7","Modelo_8","Modelo_9"] 
for modelos in range(10):
    data = numpy.concatenate((x_train_reshape, y_train_dummy[:,modelos].reshape(60000, 1)), axis=1)
    mod[modelos] = PUJ.Model.Logistic( )
    mod[modelos].setParameters( [ 0 for i in range( data.shape[ 1 ] ) ] )
   
    # Configure cost
    J = PUJ.Model.Logistic.Cost( mod[modelos], data[ : , 0 : -1 ], data[ : , -1 : ] )
    # Debugger
    debugger = PUJ.Optimizer.Debug.Simple
    
    # Fit using an optimization algorithm
    opt = PUJ.Optimizer.GradientDescent( J )
    opt.setDebugFunction( debugger )
    opt.setLearningRate( 1e-6 )
    opt.setNumberOfIterations( 500 )
    opt.setNumberOfDebugIterations( 100 )
    opt.Fit( )
    print("Terminó de entrenar el modelo: ", modelos+1 )
# ----------------------------------------------------------------------------------------
# 2.2) Ejecutar el proceso de etiquetado.
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Nombre: <<Calificacion>>
# ----------------------------------------------------------------------------------------
# Descripcion: <<Con los modelos ya entrenados, se define la función para poder etiquetar las imagenes>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
# La función necesita la data a calificar, valores de pixeles en formato Data.reshape(n, 28*28)
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna la etiqueta final del modelo para la imagen. 
# ----------------------------------------------------------------------------------------
def Calificacion(data):   
    y_prob = ["Modelo_0","Modelo_1","Modelo_2","Modelo_3","Modelo_4","Modelo_5","Modelo_6","Modelo_7","Modelo_8","Modelo_9"]
    for modelos in range(10):
        y_prob[modelos] = mod[modelos].evaluate(data)
    clients = pd.concat([pd.DataFrame(y_prob[0]), pd.DataFrame(y_prob[1]), pd.DataFrame(y_prob[2]), pd.DataFrame(y_prob[3]), pd.DataFrame(y_prob[4]), pd.DataFrame(y_prob[5]), pd.DataFrame(y_prob[6]), pd.DataFrame(y_prob[7]), pd.DataFrame(y_prob[8]), pd.DataFrame(y_prob[9])], axis=1,)
    clients.columns =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    Valor = clients.idxmax(axis = 1) 
    return Valor
# ----------------------------------------------------------------------------------------
# Nombre: <<Ejecutable>>
# ----------------------------------------------------------------------------------------
# Descripcion: <<Recibe una o varias imagenes para mostrar y clasificar según el algortimo>>
# ----------------------------------------------------------------------------------------
# PARAMETROS & PRE-CONDICIONES
# La función necesita la image a calificar, puede ser una o varias imagenes.
# ----------------------------------------------------------------------------------------
# VALOR DE RETORNO & POSTCONDICIONES
# 1. Retorna la imagen con la etiqueta que el modelo le asignó. 
# ----------------------------------------------------------------------------------------
def Ejecutable(image):
    if len(image.shape) == 3:
        image_fit = image.reshape(image.shape[0],28*28) 
        etiqueta_fit = Calificacion(image_fit)
        for l in range(image.shape[ 0 ]):
            print("El número que tiene la imagen es: ",etiqueta_fit[l])
            cv2.imshow("image", cv2.resize(image[l], (320,320)))
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = cv2.resize(image, (28,28))
        image_fit = image.reshape(1,28*28) 
        etiqueta_fit = Calificacion(image_fit)
        print("El número que tiene la imagen es: ",etiqueta_fit[0])
        cv2.imshow("image", cv2.resize(image, (320,320)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# ----------------------------------------------------------------------------------------
# <<Validación del modelo entrenado para la base de TRAIN, muestra una matriz 10x10 con todas las etiquetas 
# y el numero de casos correctos adicional muestra el porcentaje de imagenes bien clasificadas.>>
# ----------------------------------------------------------------------------------------
y_real = y_train.reshape(60000, 1)
y_est = Calificacion(x_train_reshape)
K = numpy.zeros( ( 10, 10 ) )

for i in range(y_real.shape[ 0 ]):
  K[ int( y_real[ i ] ), int( y_est[ i ] ) ] += 1
# end for
Aciertos = numpy.trace(K)/y_real.shape[ 0 ]
print("El porcentaje de imagenes bien clasificadas para train es de: ",Aciertos)
### 0.8738
# ----------------------------------------------------------------------------------------
# <<Validación del modelo entrenado para la base de TEST, muestra una matriz 10x10 con todas las etiquetas 
# y el numero de casos correctos adicional muestra el porcentaje de imagenes bien clasificadas.>>
# ----------------------------------------------------------------------------------------
y_real_test = y_test.reshape(10000, 1)
y_est_test = Calificacion(x_test_reshape)
K_Test = numpy.zeros( ( 10, 10 ) )

for i in range(y_est_test.shape[ 0 ]):
  K_Test[ int( y_real_test[ i ] ), int( y_est_test[ i ] ) ] += 1
# end for
Aciertos_Test = numpy.trace(K_Test)/y_real_test.shape[ 0 ]
print("El porcentaje de imagenes bien clasificadas para test es de: ",Aciertos_Test)
### 0.8829
# ----------------------------------------------------------------------------------------
# 2.3) Ejecución.
# ----------------------------------------------------------------------------------------
#%%
### Tomar imagen de los datos Mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image = x_test[125:130]

### Cargar imagen desde el computador
#image_2 = cv2.imread("C:/Users/User/Desktop/Prueba/7.jpg")
#image = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

Ejecutable(image)
# ----------------------------------------------------------------------------------------
# end.
# ----------------------------------------------------------------------------------------