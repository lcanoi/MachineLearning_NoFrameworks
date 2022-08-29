import pandas as pd
import numpy as np

# Input y Output Data
wineColumns = ["output",'f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']
df = pd.read_csv('data/wine.data', names=wineColumns)

input = df[['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']]
output = df['output']

## Layers y Neuronas ##
'''
capa entrada = 13 neuronas
capa oculta = 4 neuronas
capa salida = 3 neuronas
'''

one_outputs = np.zeros((len(input), 3))

for i in range(len(input)):
    one_outputs[i, output[i]-1] = 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum()

instancias = input.shape[0]
atributos = input.shape[1]
neuronas_ocultas = 4
clasificaciones = len(output.unique())

wh = np.random.rand(atributos,neuronas_ocultas)
bh = np.random.randn(neuronas_ocultas)

wo = np.random.rand(neuronas_ocultas,clasificaciones)
bo = np.random.randn(clasificaciones)
lr = 10e-5

error_cost = []

for epoch in range(10000):
    # Feedforward 
    zh = np.dot(input, wh) + bh
    ah = sigmoid(zh)

    zo = np.dot(ah, wo) + bo
    final_output = softmax(zo)

    # Backpropagation

    dcost_dzo = final_output - one_outputs
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = input
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh 
    
    # Actualizar pesos

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-one_outputs * np.log(final_output))
        print('Loss function value: ', loss)
        error_cost.append(loss)

