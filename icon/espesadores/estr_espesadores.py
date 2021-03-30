import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as we
we.filterwarnings('ignore')

def estr_espesadores(db_espesadores):
    
    """Función que estructura la data ../../../../datos/Base Datos Espesadores de Relaves 09092020.xlsx"
       en 3 archivos pickles para alimentacion, espesadores y manejo de agua.
    
    Parameters
    -------------
    
    db_epesadores: data del archivo "datos/Base Datos Espesadores de Relaves 09092020.xlsx"
    
    Returns: 
        
        data estrucuturada para datos alimentacion.
        data estrucuturada para datos de espesadores 6, 7 y 8.
        data estrucuturada para datos de manejo de agua.
    -------
    """

    # =============================================================================
    # Pre procesamiento
    # =============================================================================
    
    aux_1 = db_espesadores.iloc[:,[1,2,3,4]]
    var_title = ['fecha', 'inicio', 'termino', 'turno']
    aux_1 = aux_1.iloc[4:]
    aux_1.reset_index(inplace=True, drop=True)
    aux_1.columns = var_title
    
    # rename
    dict_name = {
        'fecha': 'Fecha',
        'inicio': 'Inicio',
        'termino': 'termino',
        'turno': 'Turno',
        }
    
    # fechas
    aux_1['fecha'] = aux_1['fecha'].astype('datetime64[ns]')
    aux_1['dia'] = aux_1['fecha'].dt.day
    aux_1['mes'] = aux_1['fecha'].dt.month
    aux_1['año'] = aux_1['fecha'].dt.year
    
    inicio = np.arange(8, 24).tolist()+np.arange(0, 8).tolist()
    inicio = inicio*1935
    
    aux_1['inicio'] = inicio
    
    aux_1['dia'][(aux_1['inicio']>=0) & (aux_1['inicio']<=7)] = aux_1['dia'] + 1
    aux_1['dia'][aux_1['dia']==32] = 1
    
    aux_1['dia'][(aux_1['mes']==4) & (aux_1['dia']==31)] = 1
    aux_1['dia'][(aux_1['mes']==6) & (aux_1['dia']==31)] = 1
    aux_1['dia'][(aux_1['mes']==9) & (aux_1['dia']==31)] = 1
    aux_1['dia'][(aux_1['mes']==11) & (aux_1['dia']==31)] = 1
    
    
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2015) &
                 (aux_1['dia']==29)] = 1
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2017) &
                 (aux_1['dia']==29)] = 1
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2018) &
                 (aux_1['dia']==29)] = 1
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2019) &
                 (aux_1['dia']==29)] = 1
    
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2016) &
                 (aux_1['dia']==30)] = 1
    aux_1['dia'][(aux_1['mes']==2) & (aux_1['año']==2020) &
                 (aux_1['dia']==30)] = 1
    
    aux_1['inicio'] = aux_1['inicio'].astype(str)
    aux_1['inicio'][aux_1['inicio']=='0'] = '00'
    aux_1['inicio'][aux_1['inicio']=='1'] = '01'
    aux_1['inicio'][aux_1['inicio']=='2'] = '02'
    aux_1['inicio'][aux_1['inicio']=='3'] = '03'
    aux_1['inicio'][aux_1['inicio']=='4'] = '04'
    aux_1['inicio'][aux_1['inicio']=='5'] = '05'
    aux_1['inicio'][aux_1['inicio']=='6'] = '06'
    aux_1['inicio'][aux_1['inicio']=='7'] = '07'
    aux_1['inicio'][aux_1['inicio']=='8'] = '08'
    aux_1['inicio'][aux_1['inicio']=='9'] = '09'
    
    aux_1['fecha_inicio'] = aux_1['año'].astype(str) + \
                            '-' + aux_1['mes'].astype(str) + \
                            '-' + aux_1['dia'].astype(str)
    
    aux_1['fecha_inicio'] = aux_1['fecha_inicio'].astype(str) + \
        ' ' + aux_1['inicio'].astype(str) + ':00:00'
    
    aux_1['fecha_inicio'] = aux_1['fecha_inicio'].astype('datetime64[ns]')
    del aux_1['inicio']; del aux_1['dia']; del aux_1['mes']; del aux_1['año']
    del aux_1['fecha']
    
    aux_1 = aux_1[['fecha_inicio', 'turno', 'termino']]
    
    # =============================================================================
    # Alimentacion Planta
    # =============================================================================
    alimentacion = db_espesadores.iloc[:,5:7]
    alimentacion = alimentacion.iloc[4:]
    vars_=['tph_sag', 'porcSolido_cajon']
    alimentacion.columns = vars_
    alimentacion.reset_index(inplace=True, drop=True)
    
    alimentacion = pd.concat([aux_1, alimentacion], axis = 1)
    
    alimentacion.sort_values('fecha_inicio', inplace=True)
    
    # rename
    dict_name = {
        'fecha': 'Fecha',
        'inicio': 'Inicio',
        'termino': 'termino',
        'turno': 'Turno',
        'tph_sag':'Tratamiento Planta SAG (TPH)',
        'porcSolido_cajon': 'Sólido Cajon  alimentación Espesadores %'
        }
    
    
    # =========================================================================
    # Espesadores 6 7 8
    # =========================================================================
    espesadores = db_espesadores.iloc[:,7:39]
    aux_title = ['6']*8 + ['7']*8 + ['8']*8 + ['total']*8
    var_title = ['alimentacion_pulpa', 'descarga_pulpa', 'n_agua_clara',
                 'floculante', 'corriente_rastras', 'torque_rastras',
                 'n_rebose_canaleta', 'descarga_solido']
    var_title = var_title*4
    espesadores = espesadores.iloc[4:]
    var_title = pd.Series(var_title).astype(str) + '_' + aux_title
    var_title = var_title.values.tolist()
    espesadores.columns = var_title
    espesadores.reset_index(inplace=True, drop=True)
    
    espesadores = pd.concat([aux_1, espesadores], axis=1)
    
    espesadores.sort_values('fecha_inicio', inplace=True)
    
    # rename
    dict_name = {
        'fecha':'Fecha',
        'inicio':'Inicio',
        'Termino':'termino',
        'turno':'Turno',
        'alimentacion_pulpa':'Alimentación Pulpa  m3/h',
        'descarga_pulpa':'Flujo descarga pulpa m3/h',
        'n_agua_clara':'Nivel Agua Clara mt',
        'floculante':'Dosificación Floculante gpt',
        'corriente_rastras':'Corriente Rastras Amper',
        'torque_rastras':'Torque Rastras %',
        'n_rebose_canaleta':'Nivel de Rebose de Canaleta %', # agua que sale
        'descarga_solido':'Sólido Descarga %',
        }
    
    # =========================================================================
    # Manejo agua
    # =========================================================================
    manejo_agua = db_espesadores.iloc[:,39:]
    var_title = ['n_cajon165', 'n_tk10', 'n_tk11', 'n_piscina50000N',
                 'n_piscina50000S', 'n_tk83']
    manejo_agua = manejo_agua.iloc[4:]
    manejo_agua.columns = var_title
    manejo_agua.reset_index(inplace=True, drop=True)
    manejo_agua = pd.concat([aux_1, manejo_agua], axis=1)
    
    manejo_agua.sort_values('fecha_inicio', inplace=True)

    # rename
    dict_name = {
        'n_cajon165':'Nivel Cajón 165',
        'n_tk10':'Nivel TK-10',
        'n_tk11':'Nivel TK-11',
        'n_piscina50000N':'Nivel Piscina 50.000 N',
        'n_piscina50000S':'Nivel Piscina 50.000 S',
        'n_tk83':'Nivel Tk-83'
        }
    
    # =========================================================================
    # Export
    # =========================================================================
    alimentacion.to_pickle('../../../../datos/alimentacion.pkl')
    espesadores.to_pickle('../../../../datos/espesadores.pkl')
    manejo_agua.to_pickle('../../../../datos/manejo_agua.pkl')
    
    print('alimentacion exportado en "../../../../datos/alimentacion.pkl"')
    print('espesadores exportado en "../../../../datos/espesadores.pkl"')
    print('manejo_agua exportado en "../../../../datos/manejo_agua.pkl"')



