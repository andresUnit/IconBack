import pandas as pd
import numpy as np
from estr_espesadores import (estr_espesadores)
from pp_espesadores import (pp_espesadores)
from sklearn.preprocessing import MinMaxScaler
from Modelo_torque import m_torque
from keras.models import load_model

# def orquestador():

# =============================================================================
# Estructuracion
# =============================================================================
db_espesadores = pd.read_excel('../../../../datos/Base Datos Espesadores de Relaves 09092020.xlsx')
estr_espesadores(db_espesadores)
del db_espesadores

# =============================================================================
# Preprocesamiento
# =============================================================================
espesadores = pd.read_pickle('../../../../datos/espesadores.pkl')
alimentacion = pd.read_pickle('../../../../datos/alimentacion.pkl')
manejo_agua = pd.read_pickle('../../../../datos/manejo_agua.pkl')

pp_espesadores(espesadores, alimentacion, manejo_agua)
del espesadores
del alimentacion
del manejo_agua

# =============================================================================
# Modelamiento
# =============================================================================
df = pd.read_pickle("../../../../datos/espesadores_7.pkl")
scaler = MinMaxScaler(feature_range=(0, 1))
tipos = ['descarga_solido_7', 'n_agua_clara_7', 'torque_rastras_7']

ventana = 1
salida = 1

# entrenamiento
m_torque(df, ventana, salida, scaler, tipos, epoch = 2)

# =============================================================================
# Simulacion
# =============================================================================
model = load_model("modelos/descarga_torque.h5")
