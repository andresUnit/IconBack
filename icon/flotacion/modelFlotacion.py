# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:14:51 2021

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

df = pd.read_excel("dataflotacion.xlsx")
df["density"]= 1/(df["%Sol Alimen. Rougher"]*0.01/2.75+(1-df["%Sol Alimen. Rougher"]*0.01))
df["caudal"] = df["Tph tratamiento"]*df["density"]*df["%Sol Alimen. Rougher"]
df["filas"] = 8
df["celdas"] = 9
df["Volumen util"] = df["filas"]* df["celdas"]*300*0.85
df["Tiempo residencia"] = df["Volumen util"]/(df["caudal"]/df["filas"])

variables_rec = ["Recuperacion Global","%Cu Conc final","%Cu Cola final","%Cu Alimen. Rougher","%Fe Alimen. Rougher","%Sol Alimen. Rougher","%Cu colas Rougher","%Fe colas Rougher","%Sol colas Rougher","%Cu Conc. Rougher","%Cu Concentrado limpieza Rougher","%FeConcentrado limpieza Rougher","%Sol Concentrado limpieza Rougher", "DI-101", "Espumante STD", "Xantato", "NaHS", "PH Rougher", "filas", "celdas", "Tiempo residencia", "Tph tratamiento"]

variables_Cuf = ["%Cu Conc final","Recuperacion Global","%Cu Cola final","%Cu Alimen. Rougher","%Fe Alimen. Rougher","%Sol Alimen. Rougher","%Cu colas Rougher","%Fe colas Rougher","%Sol colas Rougher","%Cu Conc. Rougher","%Cu Concentrado limpieza Rougher","%FeConcentrado limpieza Rougher","%Sol Concentrado limpieza Rougher", "DI-101", "Espumante STD", "Xantato", "NaHS", "PH Rougher", "filas", "celdas", "Tiempo residencia", "Tph tratamiento"]

variables_Cuc = ["%Cu Cola final","Recuperacion Global","%Cu Conc final","%Cu Alimen. Rougher","%Fe Alimen. Rougher","%Sol Alimen. Rougher","%Cu colas Rougher","%Fe colas Rougher","%Sol colas Rougher","%Cu Conc. Rougher","%Cu Concentrado limpieza Rougher","%FeConcentrado limpieza Rougher","%Sol Concentrado limpieza Rougher", "DI-101", "Espumante STD", "Xantato", "NaHS", "PH Rougher", "filas", "celdas", "Tiempo residencia", "Tph tratamiento"]

data = df[variables_rec]

# data_Fe = df[variables_Fe]
descriptivo_basico = data.describe(percentiles=[.01,.05, 0.1, .25,.5,.75,.90, .95, .99])
scaler = MinMaxScaler(feature_range=(0, 1))

cotas= descriptivo_basico.loc[["1%","99%"]]

datamin = data[data> cotas.loc["1%"]]
datamin.fillna(value = cotas.loc["1%"], inplace=True)
datamin = datamin[datamin< cotas.loc["99%"]]
datamin.fillna(value = cotas.loc["99%"], inplace=True)

descriptivo_basico2 = datamin.describe(percentiles=[.01,.05, 0.1, .25,.5,.75,.90, .95, .99])

#data[data> cotas.loc["99%"]] = cotas.loc["99%"]

#datafilter = data[(data > cotas.loc["1%"])&(data <cotas.loc["99%"])]

dataclean = datamin

datacleanCuf = dataclean.reindex(columns=variables_Cuf)
datacleanCuc = dataclean.reindex(columns=variables_Cuc)
scaler_Cuf = MinMaxScaler(feature_range=(0, 1))
scaler_Cuc = MinMaxScaler(feature_range=(0, 1))
# descriptivo_basico_fe = data_Fe.describe(percentiles=[.01,.05, 0.1, .25,.5,.75,.90, .95, .99])
# scaler_fe = MinMaxScaler(feature_range=(0, 1))

# cotas_fe= descriptivo_basico_fe.loc[["1%","99%"]]
# datafilter_fe = data_Fe[(data_Fe > cotas_fe.loc["1%"])&(data_Fe <cotas_fe.loc["99%"])]
# dataclean_fe = datafilter_fe.dropna(how="any")



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
    values = df.values
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
    print("Recuperacion Global: ", vector[0][1])
    print("%Cu Conc final:", vector[0][0])
    print("%Cu Cola final:", vector[0][2])
    print("%Cu Alimen. Rougher:", vector[0][3])
    print("%Fe Alimen. Rougher:", vector[0][4])
    print("DI-101: ", vector[0][13])
    print("Espumante STD: ", vector[0][14])
    print("Xantato: ", vector[0][15])
    print("NaHS: ", vector[0][16])
    print("PH Rougher: ", vector[0][17])
    print("Razon Cu/Fe: ", vector[0][3]/vector[0][4])
    print("Tiempo Residencia: ", vector[0][20]*60)
    print("TPH: ", vector[0][21])
    print("Filas: ", vector[0][18])
    print("celdas: ", vector[0][19])
    


ventana = 1
salida = 1


# scaled = scale_data(dataclean)
# reframed, n_vars = data_window(scaled, ventana, salida)

# train_X, train_y, test_X, test_y = data_split(reframed, ventana, n_vars, 0.9)
# train_X_reshape = data_reshape(train_X, ventana , n_vars)
# test_X_reshape = data_reshape(test_X, ventana , n_vars)


# model = Sequential()
# model.add(LSTM(50, input_shape=(ventana, n_vars)))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network

# train_model(model,train_X_reshape, train_y, test_X_reshape, test_y, epoch = 50)

# model.save("D:\GIT\ICON\Modelos\modelo_flotacion_fix.h5")

scaled_Cuf = scale_data(datacleanCuf, scaler = scaler_Cuf)
reframed_Cuf, n_vars_Cuf = data_window(scaled_Cuf, ventana, salida)

train_X_Cuf, train_y_Cuf, test_X_Cuf, test_y_Cuf = data_split(reframed_Cuf, ventana, n_vars_Cuf, 0.9)
train_X_reshape_Cuf = data_reshape(train_X_Cuf, ventana , n_vars_Cuf)
test_X_reshape_Cuf = data_reshape(test_X_Cuf, ventana , n_vars_Cuf)


model_Cuf = Sequential()
model_Cuf.add(LSTM(50, input_shape=(ventana, n_vars_Cuf)))
model_Cuf.add(Dense(1))
model_Cuf.compile(loss='mae', optimizer='adam')
# fit network

train_model(model_Cuf,train_X_reshape_Cuf, train_y_Cuf, test_X_reshape_Cuf, test_y_Cuf, epoch = 50)

model_Cuf.save("D:\GIT\ICON\Modelos\modelo_flotacion_fix_Cuf.h5")


scaled_Cuc = scale_data(datacleanCuc, scaler = scaler_Cuc)
reframed_Cuc, n_vars_Cuc = data_window(scaled_Cuc, ventana, salida)

train_X_Cuc, train_y_Cuc, test_X_Cuc, test_y_Cuc = data_split(reframed_Cuc, ventana, n_vars_Cuc, 0.9)
train_X_reshape_Cuc = data_reshape(train_X_Cuc, ventana , n_vars_Cuc)
test_X_reshape_Cuc = data_reshape(test_X_Cuc, ventana , n_vars_Cuc)


model_Cuc = Sequential()
model_Cuc.add(LSTM(50, input_shape=(ventana, n_vars_Cuc)))
model_Cuc.add(Dense(1))
model_Cuc.compile(loss='mae', optimizer='adam')
# fit network

train_model(model_Cuc,train_X_reshape_Cuc, train_y_Cuc, test_X_reshape_Cuc, test_y_Cuc, epoch = 50)

model_Cuf.save("D:\GIT\ICON\Modelos\modelo_flotacion_fix_Cuc.h5")




ini_data = np.mean(test_X_Cuf, axis= 0)


def simulacion(state, change = False, falla = False):
    if change:
        print("Estado Inicial")
        
        state_ini_rescaled = scaler_Cuf.inverse_transform(state.reshape(1,-1))
        var_rec_aux = state_ini_rescaled[0][1]
        print_data(state_ini_rescaled)
        elemento = str(input("Ingrese la Variable (tph, cu, fe,di,esp,xan,na,ph): "))
        print(elemento)
        cantidad = input("Ingrese Cantidad: ")
        elementos = ["cu","fe","di", "esp", "xan", "na", "ph"]
        if elemento=="tph":
            index = 21
        elif elemento=="fe":
            index = 4
        elif elemento=="cu":
            index = 3
        else:
            index = elementos.index(elemento)+11
        if falla:
            fallas = input("Ingrese la Variable (fi, ce): ")
            cantidades = input("Ingrese Cantidad: ")
            if fallas == "fi":
                state_ini_rescaled[0][18]=float(cantidades)
            elif fallas == "ce":
                state_ini_rescaled[0][19]=float(cantidades)
        state_ini_rescaled[0][index]=float(cantidad)
        density = 1/(state_ini_rescaled[0][5]*0.01/2.75+(1-state_ini_rescaled[0][5]*0.01))
        caudal = state_ini_rescaled[0][21]*density*state_ini_rescaled[0][5]/state_ini_rescaled[0][18]
        Volumen = state_ini_rescaled[0][18]*state_ini_rescaled[0][19]*300*0.85
        state_ini_rescaled[0][20] = Volumen/caudal
        state_aux = scaler_Cuf.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        state_aux_Cuc = state_aux.copy()
        state_aux_Cuc[[2, 0]] = state_aux_Cuc[[0, 2]]
        pred_Cuf = model_Cuf.predict(state_aux.reshape((1,1,-1)))
        pred_Cuc = model_Cuc.predict(state_aux_Cuc.reshape((1,1,-1)))
        variacion_Cuf = (pred_Cuf-state_aux[0])*100/state_aux[0]
        variacion_Cuc = (pred_Cuc-state_aux[2])*100/state_aux[2]
        state_aux[0] = pred_Cuf
        state_aux[2] = pred_Cuc
        print("Estado Final")
        state_fin_rescaled = scaler_Cuf.inverse_transform(state_aux.reshape(1,-1))
        state_fin_rescaled[0][1]=((state_fin_rescaled[0][3]-state_fin_rescaled[0][2])*state_fin_rescaled[0][0])/((state_fin_rescaled[0][0]-state_fin_rescaled[0][2])*state_fin_rescaled[0][3])*100
        variacion_rec = (state_fin_rescaled[0][1]-var_rec_aux)*100/var_rec_aux
        print_data(state_fin_rescaled)
        print("Variacion de Recuperacion: ")
        print(variacion_rec)
        print("Variacion de Ley de Concentrado: ")
        print(variacion_Cuf[0][0])
        print("Variacion de Ley de Cola Final: ")
        print(variacion_Cuc[0][0])
        return state_aux
        
    else:
        print("Estado Inicial")
        state_ini_rescaled = scaler_Cuf.inverse_transform(state.reshape(1,-1))
        var_rec_aux = state_ini_rescaled[0][1]
        print_data(state_ini_rescaled)
        density = 1/(state_ini_rescaled[0][5]*0.01/2.75+(1-state_ini_rescaled[0][5]*0.01))
        caudal = state_ini_rescaled[0][21]*density*state_ini_rescaled[0][5]/state_ini_rescaled[0][18]
        Volumen = state_ini_rescaled[0][18]*state_ini_rescaled[0][19]*300*0.85
        state_ini_rescaled[0][20] = Volumen/caudal
        state_aux = scaler_Cuf.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        state_aux_Cuc = state_aux.copy()
        state_aux_Cuc[[2, 0]] = state_aux_Cuc[[0, 2]]
        pred_Cuf = model_Cuf.predict(state_aux.reshape((1,1,-1)))
        pred_Cuc = model_Cuc.predict(state_aux_Cuc.reshape((1,1,-1)))
        variacion_Cuf = (pred_Cuf-state_aux[0])*100/state_aux[0]
        variacion_Cuc = (pred_Cuc-state_aux[2])*100/state_aux[2]
        state_aux[0] = pred_Cuf
        state_aux[2] = pred_Cuc
        print("Estado Final")
        state_fin_rescaled = scaler_Cuf.inverse_transform(state_aux.reshape(1,-1))
        state_fin_rescaled[0][1]=((state_fin_rescaled[0][3]-state_fin_rescaled[0][2])*state_fin_rescaled[0][0])/((state_fin_rescaled[0][0]-state_fin_rescaled[0][2])*state_fin_rescaled[0][3])*100
        variacion_rec = (state_fin_rescaled[0][1]-var_rec_aux)*100/var_rec_aux
        print(var_rec_aux)
        print_data(state_fin_rescaled)
        print("Variacion de Recuperacion: ")
        print(variacion_rec)
        print("Variacion de Ley de Concentrado: ")
        print(variacion_Cuf[0][0])
        print("Variacion de Ley de Cola Final: ")
        print(variacion_Cuc[0][0])

    return state_aux


ph = np.linspace(9.0, 11.0, num=102)

def recorrido(ini_state, variable):
    result_rec = []
    result_Cuf = []
    result_Cuc = []
    for i in variable:
        state_ini_rescaled = scaler_Cuf.inverse_transform(ini_state.reshape(1,-1))
        state_ini_rescaled[0][17]=float(i)
        state_aux = scaler_Cuf.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        state_aux_Cuc = state_aux.copy()
        state_aux_Cuc[[2, 0]] = state_aux_Cuc[[0, 2]]
        pred_Cuf = model_Cuf.predict(state_aux.reshape((1,1,-1)))
        pred_Cuc = model_Cuc.predict(state_aux_Cuc.reshape((1,1,-1)))
        state_aux[0] = pred_Cuf
        state_aux[2] = pred_Cuc
        state_fin_rescaled = scaler_Cuf.inverse_transform(state_aux.reshape(1,-1))
        state_fin_rescaled[0][1]=((state_fin_rescaled[0][3]-state_fin_rescaled[0][2])*state_fin_rescaled[0][0])/((state_fin_rescaled[0][0]-state_fin_rescaled[0][2])*state_fin_rescaled[0][3])*100
        result_rec.append(state_fin_rescaled[0][1])
        result_Cuf.append(state_fin_rescaled[0][0])
        result_Cuc.append(state_fin_rescaled[0][2])
    return result_rec, result_Cuf, result_Cuc

ini_data_aux = ini_data.copy()

ini_data_aux[3] = float(0.6)

result_rec, result_Cuf, result_Cuc = recorrido(ini_data_aux, ph)

plt.plot(ph, result_rec)
plt.plot(ph, result_Cuf)
plt.plot(ph, result_Cuc)

