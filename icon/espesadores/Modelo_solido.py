# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:52:05 2021

author: AMS
"""

import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model

df = pd.read_pickle("espesadores_7.pkl")

scaler = MinMaxScaler(feature_range=(0, 1))

tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

# def Data_base(df,tipo):
#     df.fillna(df.mean(), inplace=True)
#     data =  [f'{tipo}','alimentacion_pulpa_7','descarga_pulpa_7','floculante_7','porcSolido_cajon']
#     df_aux = df[data]
#     return df_aux

# def scale_data(df, scaler = scaler):
#     df = df.astype('float32')
#     scaled = scaler.fit_transform(df)
#     return scaled


# def data_window(data, ventana = 1, salida= 1):
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     for i in range(ventana, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#     for i in range(0, salida):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#     new = pd.concat(cols, axis=1)
#     new.columns = names
#     new.dropna(inplace=True)
#     return new, n_vars


def data_split(df, ventana, n_vars, percentage):
    values = reframed.values
    n_train_hours = int(values.shape[0]*percentage)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    n_obs = ventana * n_vars
    train_X, train_y = train[:, :n_obs], train[:, -n_vars]
    test_X, test_y = test[:, :n_obs], test[:, -n_vars]
    
    return train_X, train_y, test_X, test_y

def data_reshape(df, ventana, n_vars):
    df = df.reshape((df.shape[0], ventana, n_vars))
    return df

def inv_reshape(df):
    df = df.reshape((df.shape[0], df.shape[2]))
    return df

def train_model(model, train_X, train_y, test_X, test_y, epoch = 50):
    history = model.fit(train_X, train_y, epochs=epoch, batch_size=72,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
def re_scale(yhat,test_X, scaler = scaler):
    inv_yhat = concatenate((yhat, test_X[:,1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    return inv_yhat

def print_data(vector):
    print("Descarga de solido: ", vector[0][0])
    print("Alimentacion pulpa: ", vector[0][1])
    print("Descarga de Pulpa: ", vector[0][2])
    print("Floculante: ", vector[0][3])
    print("Porcentaje de Solido: ", vector[0][4])





ventana = 1
salida = 1


#tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

data = Data_base(df,tipos[0])



scaled = scale_data(data)
reframed, n_vars = data_window(scaled, ventana, salida)
train_X, train_y, test_X, test_y = data_split(reframed, ventana, n_vars, 0.9)
train_X_reshape = data_reshape(train_X, ventana , n_vars)
test_X_reshape = data_reshape(test_X, ventana , n_vars)


model = Sequential()
model.add(LSTM(50, input_shape=(ventana, n_vars)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network

train_model(model,train_X_reshape, train_y, test_X_reshape, test_y, epoch = 50)

model.save("D:\GIT\ICON\Modelos\descarga_solido.h5")

model = load_model("D:\GIT\ICON\Modelos\descarga_solido.h5")

ini_data = np.mean(test_X, axis= 0)


def simulacion(state, model = model, change = False ):
    if change:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        elemento = input("Ingrese la Variable (ap,dp,fl,ps): ")
        cantidad = input("Ingrese Cantidad: ")
        elementos = ["ap", "dp", "fl", "ps"]
        index = elementos.index(elemento)+1
        state_ini_rescaled[0][index]=float(cantidad)
        state_aux = scaler.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        pred = model.predict(state_aux.reshape((1,1,-1)))
        variacion =(pred-state_aux[0])*100/state_aux[0]
        state_aux[0]=pred
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state_aux.reshape(1,-1))
        print_data(state_fin_rescaled)
        density = 1/(state_fin_rescaled[0][4]/2.75-(1-state_fin_rescaled[0][4]))
        flujoM = state_fin_rescaled[0][1]*density*state_fin_rescaled[0][4]
        print("Flujo Masico: ")
        print(flujoM)
        print("Variacion de Descarga de Solido: ")
        print(variacion[0][0])
        return state_aux
        
    else:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        pred = model.predict(state.reshape((1,1,-1)))
        variacion = (pred-state[0])*100/state[0]
        state[0]=pred
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_fin_rescaled)
        density = 1/(state_fin_rescaled[0][4]/2.75-(1-state_fin_rescaled[0][4]))
        flujoM = state_fin_rescaled[0][1]*density*state_fin_rescaled[0][4]
        print("Flujo Masico: ")
        print(flujoM)
        print("Variacion de Descarga de Solido: ")
        print(variacion[0][0])
    return state

