# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:33:17 2021

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

tipos = ['descarga_solido_7', 'n_agua_clara_7','torque_rastras_7' ]

def Data_base(df,tipos):
    df.fillna(df.mean(), inplace=True)
    solido = tipos[0]
    agua = tipos[1]
    torque = tipos[2]
    data =  [f'{solido}',f'{agua}',f'{torque}','alimentacion_pulpa_7','descarga_pulpa_7','floculante_7','porcSolido_cajon']
    df_aux = df[data]
    return df_aux

def scale_data(df, scaler = scaler):
    df = df.astype('float32')
    scaled = scaler.fit_transform(df)
    return scaled


def data_window(data, ventana = 1, salida= 1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(ventana, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, salida):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    new = pd.concat(cols, axis=1)
    new.columns = names
    new.dropna(inplace=True)
    return new, n_vars


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
    print("Nivel de Agua: ", vector[0][1])
    print("Torque: ", vector[0][2])
    print("Alimentacion pulpa: ", vector[0][3])
    print("Descarga de Pulpa: ", vector[0][4])
    print("Floculante: ", vector[0][5])
    print("Porcentaje de Solido: ", vector[0][6])
    
    





ventana = 1
salida = 1


#tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

data = Data_base(df,tipos)



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

model.save("D:\GIT\ICON\Modelos\descarga_torque_fix.h5")

model_solido = load_model("D:\GIT\ICON\Modelos\descarga_solido_fix.h5")
model_agua = load_model("D:\GIT\ICON\Modelos\descarga_agua_fix.h5")
model_torque = load_model("D:\GIT\ICON\Modelos\descarga_torque_fix.h5")
ini_data = np.mean(test_X, axis= 0)


def simulacion(state, change = False ):
    if change:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        elemento = input("Ingrese la Variable (ap,dp,fl,ps): ")
        cantidad = input("Ingrese Cantidad: ")
        elementos = ["ap", "dp", "fl", "ps"]
        index = elementos.index(elemento)+3
        state_ini_rescaled[0][index]=float(cantidad)
        state_aux = scaler.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        state_aux_agua = np.array([state_aux[1],state_aux[0], state_aux[2],state_aux[3],state_aux[4],state_aux[5], state_aux[6]])
        state_aux_torque = np.array([state_aux[2],state_aux[1], state_aux[0],state_aux[3],state_aux[4],state_aux[5], state_aux[6]])
        pred_solido = model_solido.predict(state_aux.reshape((1,1,-1)))
        pred_agua = model_agua.predict(state_aux_agua.reshape((1,1,-1)))
        pred_torque = model_torque.predict(state_aux_torque.reshape((1,1,-1)))
        variacion_solido = (pred_solido-state_aux[0])*100/state_aux[0]
        variacion_agua = (pred_agua-state_aux[1])*100/state_aux[1]
        variacion_torque = (pred_torque-state_aux[2])*100/state_aux[2]
        state_aux[0]=pred_solido
        state_aux[1]=pred_agua
        state_aux[2]=pred_torque
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state_aux.reshape(1,-1))
        print_data(state_fin_rescaled)
        print("Variacion de Descarga de Solido: ")
        print(variacion_solido[0][0])
        print("Variacion de Nivel de Agua: ")
        print(variacion_agua[0][0])
        print("Variacion de Torque: ")
        print(variacion_torque[0][0])
        density = 1/(state_fin_rescaled[0][6]*0.01/2.75+(1-state_fin_rescaled[0][6]*0.01))
        flujoM = state_fin_rescaled[0][3]*density*state_fin_rescaled[0][6]*0.01
        flujoS = state_fin_rescaled[0][4]*density*state_fin_rescaled[0][6]*0.01
        print(" Delta Flujo Masico: ")
        print(flujoM-flujoS)
        return state_aux
        
    else:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        state_aux = scaler.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        state_aux_agua = np.array([state_aux[1],state_aux[0], state_aux[2],state_aux[3],state_aux[4],state_aux[5], state_aux[6]])
        state_aux_torque = np.array([state_aux[2],state_aux[1], state_aux[0],state_aux[3],state_aux[4],state_aux[5], state_aux[6]])
        pred_solido = model_solido.predict(state_aux.reshape((1,1,-1)))
        pred_agua = model_agua.predict(state_aux_agua.reshape((1,1,-1)))
        pred_torque = model_torque.predict(state_aux_torque.reshape((1,1,-1)))
        variacion_solido = (pred_solido-state_aux[0])*100/state_aux[0]
        variacion_agua = (pred_agua-state_aux[1])*100/state_aux[1]
        variacion_torque = (pred_torque-state_aux[2])*100/state_aux[2]
        state_aux[0]=pred_solido
        state_aux[1]=pred_agua
        state_aux[2]=pred_torque
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state_aux.reshape(1,-1))
        print_data(state_fin_rescaled)
        print("Variacion de Descarga de Solido: ")
        print(variacion_solido[0][0])
        print("Variacion de Nivel de Agua: ")
        print(variacion_agua[0][0])
        print("Variacion de Torque: ")
        print(variacion_torque[0][0])
        density = 1/(state_fin_rescaled[0][6]*0.01/2.75+(1-state_fin_rescaled[0][6]*0.01))
        print(density)
        flujoM = state_fin_rescaled[0][3]*density*state_fin_rescaled[0][6]*0.01
        print(flujoM)
        flujoS = state_fin_rescaled[0][4]*density*state_fin_rescaled[0][6]*0.01
        print(flujoS)
        print("Delta Flujo Masico: ")
        print(flujoM-flujoS)
    return state_aux
    












