import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from funciones.stats_code import *
from funciones.f_series import *
import warnings as we
we.filterwarnings('ignore')
pd.set_option('max.columns', 100)

def pp_espesadores(espesadores, alimentacion, manejo_agua):
    
    """Función que realiza un preprocesamiento de los datos de alimentacion y los
    datos de los espesadores.
    
    Parameters
    -------------
    
    espesadores: datos en formato pickle de los espesadores.
    alimentacion: datos en format pickle de la alimentacion.
    manejo_agua: datos en formato pickle del manejo de agua.
    
    Returns: 

    -------
    """

    del espesadores['turno']
    del espesadores['termino']
    espesadores.set_index('fecha_inicio', inplace=True)
    alimentacion.set_index('fecha_inicio', inplace=True)
    
    espesadores = espesadores.astype(float)
    
    espesadores = pd.concat([espesadores, alimentacion[['tph_sag', 'porcSolido_cajon']]],
                            axis=1)
    
    # =============================================================================
    # calculo variables nuevas
    # =============================================================================
    
    # entrada
    espesadores['densidad_pulpa_e_6'] = 1/((espesadores['porcSolido_cajon']*0.01/2.75)+\
                                         (1-espesadores['porcSolido_cajon']*0.01))
    
    espesadores['flujo_masico_e_6'] = espesadores['porcSolido_cajon']*0.01*\
                                           espesadores['alimentacion_pulpa_6']*\
                                           espesadores['densidad_pulpa_e_6']
    
    
    espesadores['densidad_pulpa_e_7'] = 1/((espesadores['porcSolido_cajon']*0.01/2.75)+\
                                         (1-espesadores['porcSolido_cajon']*0.01))
    
    espesadores['flujo_masico_e_7'] = espesadores['porcSolido_cajon']*0.01*\
                                           espesadores['alimentacion_pulpa_7']*\
                                           espesadores['densidad_pulpa_e_7']
    
    
    espesadores['densidad_pulpa_e_8'] = 1/((espesadores['porcSolido_cajon']*0.01/2.75)+\
                                         (1-espesadores['porcSolido_cajon']*0.01))
    
    espesadores['flujo_masico_e_8'] = espesadores['porcSolido_cajon']*0.01*\
                                           espesadores['alimentacion_pulpa_8']*\
                                           espesadores['densidad_pulpa_e_8']
    
    # salida
    espesadores['densidad_pulpa_s_6'] = 1/((espesadores['descarga_solido_6']*0.01/2.75)+\
                                         (1-espesadores['descarga_solido_6']*0.01))
    
    espesadores['flujo_masico_s_6'] = espesadores['descarga_solido_6']*0.01*\
                                           espesadores['alimentacion_pulpa_6']*\
                                           espesadores['densidad_pulpa_s_6']
    
    espesadores['densidad_pulpa_s_7'] = 1/((espesadores['descarga_solido_7']*0.01/2.75)+\
                                         (1-espesadores['descarga_solido_7']*0.01))
    
    espesadores['flujo_masico_s_7'] = espesadores['descarga_solido_7']*0.01*\
                                           espesadores['alimentacion_pulpa_7']*\
                                           espesadores['densidad_pulpa_s_7']
    
    espesadores['densidad_pulpa_s_8'] = 1/((espesadores['descarga_solido_8']*0.01/2.75)+\
                                         (1-espesadores['descarga_solido_8']*0.01))
    
    espesadores['flujo_masico_s_8'] = espesadores['descarga_solido_8']*0.01*\
                                           espesadores['alimentacion_pulpa_8']*\
                                           espesadores['densidad_pulpa_s_8']
    
    # save
    espesadores.to_pickle('../../../../datos/espesadores1.pkl')
    print('espesadores exportado en "../../../../datos/espesadores1.pkl"')
    
    # =============================================================================
    # Division espesadores 6 7 8
    # =============================================================================
    
    espesadores_6 = espesadores[['alimentacion_pulpa_6', 'descarga_pulpa_6',
                                'n_agua_clara_6', 'floculante_6', 'corriente_rastras_6',
                                'torque_rastras_6', 'n_rebose_canaleta_6',
                                'descarga_solido_6',
                                'densidad_pulpa_e_6','densidad_pulpa_s_6',
                                'flujo_masico_e_6', 'flujo_masico_s_6',
                                'tph_sag', 'porcSolido_cajon']]
    espesadores_6 = espesadores_6.astype(float)
    
    espesadores_7 = espesadores[['alimentacion_pulpa_7', 'descarga_pulpa_7',
                                'n_agua_clara_7', 'floculante_7', 'corriente_rastras_7',
                                'torque_rastras_7', 'n_rebose_canaleta_7',
                                'descarga_solido_7',
                                'densidad_pulpa_e_7','densidad_pulpa_s_7',
                                'flujo_masico_e_7', 'flujo_masico_s_7',
                                'tph_sag', 'porcSolido_cajon']]
    espesadores_7 = espesadores_7.astype(float)
    
    espesadores_8 = espesadores[['alimentacion_pulpa_8', 'descarga_pulpa_8',
                                'n_agua_clara_8', 'floculante_8', 'corriente_rastras_8',
                                'torque_rastras_8', 'n_rebose_canaleta_8',
                                'descarga_solido_8',
                                'densidad_pulpa_e_8','densidad_pulpa_s_8',
                                'flujo_masico_e_8', 'flujo_masico_s_8',
                                'tph_sag', 'porcSolido_cajon']]
    espesadores_8 = espesadores_8.astype(float)
    del espesadores
    
    # =============================================================================
    # Eliminacion datos perdidos flujo descarga y alimentacion
    # =============================================================================
    
    # %solido descarga bajo 42 fuera
    espesadores_6 = espesadores_6[espesadores_6['descarga_solido_6']>=42]
    espesadores_7 = espesadores_7[espesadores_7['descarga_solido_7']>=42]
    espesadores_8 = espesadores_8[espesadores_8['descarga_solido_8']>=42]
    
    espesadores_6 = espesadores_6[~espesadores_6['flujo_masico_s_6'].isnull()]
    espesadores_6 = espesadores_6[~espesadores_6['flujo_masico_e_6'].isnull()]
    
    espesadores_7 = espesadores_7[~espesadores_7['flujo_masico_s_7'].isnull()]
    espesadores_7 = espesadores_7[~espesadores_7['flujo_masico_e_7'].isnull()]
    
    espesadores_8 = espesadores_8[~espesadores_8['flujo_masico_s_8'].isnull()]
    espesadores_8 = espesadores_8[~espesadores_8['flujo_masico_e_8'].isnull()]
    
    # save
    espesadores_6.to_pickle('../../../../datos/espesadores_6.pkl')
    espesadores_7.to_pickle('../../../../datos/espesadores_7.pkl')
    espesadores_8.to_pickle('../../../../datos/espesadores_8.pkl')
    print('espesadores 6 exportado en "../../../../datos/espesadores_6.pkl"')
    print('espesadores 7 exportado en "../../../../datos/espesadores_7.pkl"')
    print('espesadores 8 exportado en "../../../../datos/espesadores_8.pkl"')



