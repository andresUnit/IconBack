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
import h5py

# df = pd.read_pickle("../../../../datos/espesadores_7.pkl")

# scaler = MinMaxScaler(feature_range=(0, 1))
# tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

def Data_base(df, tipo):
    
    """Función que completa los datos nulos por la media y obtiene un df solo con
    las columnas de torque y las definidas en el argumento tipos.
    
    Parameters
    -------------
    
    df: dataframe en formato pandas.
    tipo: variables que se quieren mantener 
    
    Returns: 
        
        ADS en formato dataframe (Pandas).
    -------
    """
    
    df.fillna(df.mean(), inplace=True)
    data =  [f'{tipo}','alimentacion_pulpa_7','descarga_pulpa_7','floculante_7','porcSolido_cajon']
    df_aux = df[data]
    return df_aux

def scale_data(df, scaler):
    
    """Función que realiza una transformacion como una escalamiento MinMaxScaler a los datos.

    Parameters
    -------------
    
    df: Matrix en formato Numpy.
    scaler: tipo de transformacion de la libreria sklearn.preprocessing.

    Returns: 
        
        Objeto Matrix en formato Numpy.
    -------
    """

    df = df.astype('float32')
    scaled = scaler.fit_transform(df)
    return scaled


def data_window(data, ventana = 1, salida= 1):
    
    """Función que crea un dataframe de datos con las ventanas que usa el modelo
    LSTM.

    Parameters
    -------------
    
    data: dataframe en formato pandas.
    ventana: cantidad de rezagos para crear la vetana.
    salida: 
    
    Returns: 
        
        Tupla que contiene un ADS en formato dataframe (Pandas) y el numero
        de columnas tratadas en formato int.
    -------
    """
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


def data_split(reframed, ventana, n_vars, percentage):
    
    """Función que divide los datos en la data train, test.

    Parameters
    -------------
    
    reframed: df con las columnas con rezagos correspondientes a la ventana.
    ventana: numero de rezagos de reframed.
    n_vars: numero de columnas que tiene rezago.
    percentage: porcentaje de la data que correspondera a los datos train.
    
    Returns: 
        
        ADS en formato dataframe (Pandas).
    -------
    """
    
    values = reframed.values
    n_train_hours = int(values.shape[0]*percentage)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    n_obs = ventana * n_vars
    train_X, train_y = train[:, :n_obs], train[:, -n_vars]
    test_X, test_y = test[:, :n_obs], test[:, -n_vars]
    
    return train_X, train_y, test_X, test_y

def data_reshape(df, ventana, n_vars):
    """Función que transforma los datos de una matriz a un tensor.

    Parameters
    -------------
    
    df: data en formato Numpy.
    ventana: numero de rezagos de df.
    n_vars: numero de columnas que tiene rezago.

    Returns: 
        
        Tensor en formato Numpy.
    -------
    """
    df = df.reshape((df.shape[0], ventana, n_vars))
    return df

def inv_reshape(df):
    df = df.reshape((df.shape[0], df.shape[2]))
    return df

def train_model(model, train_X, train_y, test_X, test_y, epoch = 50):
    
    """Función que entrena un modelo de redes neuronales y entrega informacion
    relevante.

    Parameters
    -------------
    
    model: modelo compilado de la libreria Keras.
    train_X: tensor con las covariables de entrenamiento
    train_y: tensor con la variable respuesta de entrenamiento
    test_X: tensor con las covariables de test.
    test_y: tensor con la variable respuesta de test.
    epoch: numero de epocas de entrenamiento.

    Returns: 
        
    -------
    """
    
    history = model.fit(train_X, train_y, epochs=epoch, batch_size=72,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
# def re_scale(yhat,test_X, scaler = scaler):
#     inv_yhat = concatenate((yhat, test_X[:,1:]), axis=1)
#     inv_yhat = scaler.inverse_transform(inv_yhat)
#     inv_yhat = inv_yhat[:,0]
#     return inv_yhat

def print_data(vector):
    print("Torque: ", vector[0][0])
    print("Alimentacion pulpa: ", vector[0][1])
    print("Descarga de Pulpa: ", vector[0][2])
    print("Floculante: ", vector[0][3])
    print("Porcentaje de Solido: ", vector[0][4])
    

# ventana = 1
# salida = 1
# tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

def m_torque(df, ventana, salida, scaler,
             tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7'],
             epoch = 50):
    
    """Función que realiza el modelo (LSTM) para la variable torque.
    
    Parameters
    -------------
    df: dataframe en formato pandas.
    ventana: cantidad de rezagos para crear la vetana.
    salida: cantidad de predicciones.
    tipos: lista con el nombre de las covariables.
    epoch: epocas de entrenamiento.

    Returns: 

        No retorna nada. Exporta los modelos directamente en la ruta:
            
            "modelos/descarga_torque.h5".
    -------
    """
    # elegimos las columnas adecuadas
    data = Data_base(df, tipos[2])

    # escalamos
    scaled = scale_data(data, scaler)
    
    # separacion de la data en data train/test y conversion a un tensor
    reframed, n_vars = data_window(scaled, ventana, salida)
    train_X, train_y, test_X, test_y = data_split(reframed, ventana, n_vars, 0.9)
    train_X_reshape = data_reshape(train_X, ventana , n_vars)
    test_X_reshape = data_reshape(test_X, ventana , n_vars)
    
    # definicion del modelo de redes
    model = Sequential()
    model.add(LSTM(50, input_shape=(ventana, n_vars)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    # entrenamiento
    train_model(model,train_X_reshape, train_y, test_X_reshape, test_y, epoch=epoch)

    # save
    model.save("modelos/descarga_torque.h5")
    print('modelo exportado a modelos/descarga_torque.h5')

