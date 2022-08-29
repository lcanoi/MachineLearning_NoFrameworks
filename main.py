import pandas as pd
import numpy as np

# Input y Output Data
wineColumns = ["output",'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']
df = pd.read_csv('data/wine.data', names=wineColumns)

input = df[['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']]
output = df['output']

input = np.array(input)

output = np.array(output)
output = output.reshape(178,1)

print(input.shape)
print(output.shape)

## Layers y Neuronas ##
'''
Vamos a probar el modelo de NN de perceptron,
con una capa de entrada de 13 neuronas y una
capa de salida de 3 neuronas.

Comenzamos con este, pero es posible que se
tenga que cambiar a otra arquitectura de modelo 
para ser más acertados con el wine dataset, que
es el que está en uso.
'''

## Asignar Pesos ##
'''
Vamos a tomar 13 valores iniciales aleatorios,
el objetivo es que el modelo los modifique 
y los optimice
'''
pesos = np.array([[0.1],[0.2],[0.3],[0.4],[0.5],[0.45],
        [0.35],[0.25],[0.15],[0.5],[0.17],[0.27],[0.37]])
print(pesos.shape)

## Definir Bias y Learning Rate ##
'''
Bias = 1, pero de igual manera comenzaremos
asignandolo aleatoriamente
Lr va a comenzar con un valor pequeño, pero
es posible que tengamos que ir modificandolo
'''
bias = 0.3
lr = 10e-6

## Función de Activación ##
'''
Probaremos con sigmoide
'''
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum()

## Entrenar el modelo ##

print("Entrenando el modelo...")
for epoch in range(10000):
    ins = input

    # Feedforward 
    in_f = np.dot(ins, pesos) + bias
    out_f = sigmoid(in_f)

    # Backpropagation
    # Error de salida
    error = out_f - output

    # Formula
    x = error.sum()
    # print(x) # Revisar
    # Este print siempre está dando de resultado -167 por alguna razón

    # Calcular derivada
    derror_doutf = error
    derror_dinf = sigmoid_der(out_f)

    # Multiplicar derivadas
    der = derror_doutf * derror_dinf

    # Encontrar transpuesta de inputs
    ins = input.T
    der_f = np.dot(ins, der)

    # Actualizar pesos
    pesos -= lr * der_f

    # Actualizar bias
    for i in der:
        bias -= lr * i

## Peso y Bias final ##
print("Pesos \n",pesos)
print("Bias \n",bias)
'''
Ejemplo de output en una corrida:
Pesos 
 [[4.20678002e+01]
 [9.19027726e+00] 
 [8.03353378e+00] 
 [6.90745822e+01] 
 [3.19935408e+02] 
 [6.77186010e+00] 
 [4.72972742e+00] 
 [1.60229688e+00] 
 [4.60306368e+00] 
 [1.87689842e+01] 
 [2.93310547e+00] 
 [7.33583966e+00] 
 [1.91448548e+03]]
Bias
 [3.58341929]
Parecen hacer sentido, sin embargo hay problemas en
las predicciones en las siguientes líneas
'''

## Predicción de Valores ##
'''
Probemos 3 puntos aleatorios diferentes
Punto 1 output = 1
Punto 2 output = 2
Punto 3 output = 3
'''
punto1 = np.array([13.24,2.59,2.87,21,118,2.8,2.69,.39,1.82,4.32,1.04,2.93,735])
punto2 = np.array([12.33,1.1,2.28,16,101,2.05,1.09,.63,.41,3.27,1.25,1.67,680])
punto3 = np.array([12.84,2.96,2.61,24,101,2.32,.6,.53,.81,4.92,.89,2.15,590])
res1_1 = np.dot(punto1, pesos) + bias
res1_2 = sigmoid(res1_1)
res2_1 = np.dot(punto2, pesos) + bias
res2_2 = sigmoid(res2_1)
res3_1 = np.dot(punto3, pesos) + bias
res3_2 = sigmoid(res3_1)
print("Resultado 1: ", res1_2)
print("Resultado 2: ", res2_2)
print("Resultado 3: ", res3_2)
'''
Revisar,
Por alguna razon a pesar de tener variedad de números en los 
np.arrays de puntos y pesos, todos los resultados son = [1.]
'''

## Calcular Error ##
'''
Primero obtener resultados bien en el paso anterior, luego ya
'''
