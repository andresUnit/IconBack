import pandas as pd
import numpy as np
import seaborn as sns
import warnings as w
# from funciones.f_series import (grupos_densidad,
#                                 mean_between)
# from funciones.stats_code import (desc_grupos)
# w.filterwarnings('ignore')
# pd.set_option('max.columns', 100)
# sns.set('paper')

def g_estacionario(sag, costos={'usd/m3': 3,'usd/kwh': 0.15,'usd/ton': 2.34}):
    
    """Función que pre-procesa los datos sag y devuelve el ads (sag_estacionario)."
    
    Parameters
    -------------
    
    sag: data sag estructurada albergado en 'datos/sag.pkl'.
    
    costos: dict. Costos del agua, potencia y toneladas.
    
        por default: {
        'usd/m3': 3,
        'usd/kwh': 0.15,
        'usd/ton': 2.34
        }
        
        *Son los costos dado en el archivo "../../datos/KCD . KVD Modelo MoliendaSAG.xlsx"
        
    Returns: 
        
        Reporte de costos.
        ADS en formato dataframe (Pandas).
    -------
    """

    # =============================================================================
    # del outlier y calculo de e_especifica_m
    # =============================================================================
    fechas_dato_anomalos = {}
    # dato anomalos potencia
    sag['pot_esc'] = sag['potencia'] / sag['potencia'].max()
    fechas_dato_anomalos['falla_presion'] = sag['potencia']\
                        [sag['potencia'] > 100000].index
    sag['potencia'][sag['potencia'] > 100000] = np.nan
    sag['potencia'] = mean_between(sag['potencia'])
    print('Dato anomalo Potencia: inputada por la media')
    
    sag = sag.astype(float)
    sag['e_especifica_m'] = sag['potencia']/sag['tph']
    sag['agua_especifica'] = sag['agua']/sag['tph']
    print('Creación de variables: e_especifica_m, agua_especifica')
    
    # =============================================================================
    # Rangos experto
    # =============================================================================
    sag['g_experto'] = -1
    sag['g_experto'][(sag['potencia'] > 16500) & (sag['potencia'] < 18000) &
                     (sag['velocidad'] > 8) & (sag['velocidad'] < 8.8) &
                     (sag['tph'] > 4000) & (sag['tph'] < 6250) &
                     (sag['gruesos'] > 10) & (sag['gruesos'] < 20) &
                     (sag['porcSolido'] > 72) & (sag['porcSolido'] < 76)
                     ] = 0

    sag['g_experto'][(sag['potencia'] > 0) & (sag['potencia'] < 16500) &
                     (sag['velocidad'] > 7) & (sag['velocidad'] < 8) &
                     (sag['tph'] > 3400) & (sag['tph'] < 4000) &
                     (sag['gruesos'] > 5) & (sag['gruesos'] < 10) &
                     (sag['porcSolido'] > 70) & (sag['porcSolido'] < 72)
                     ] = 1
        
    sag['g_experto'][(sag['potencia'] > 18000) & (sag['potencia'] < 19400) &
                     (sag['velocidad'] > 8.8) & (sag['velocidad'] < 9.2) &
                     (sag['tph'] > 6250) & (sag['tph'] < 6500) &
                     (sag['gruesos'] > 20) & (sag['gruesos'] < 25) &
                     (sag['porcSolido'] > 76) & (sag['porcSolido'] < 80)
                     ] = 2
    print('Creación variable rango experto: g_experto')

    # =============================================================================
    # Deteccion de molino detenido
    # =============================================================================
    
    sag['detenido'] = 0
    sag['detenido'][sag['velocidad'] < 1] = 1
    # =============================================================================
    # Deteccion de ingreso de mineral
    # =============================================================================
    
    sag['nivel_ton'] = 99
    sag['nivel_ton'][sag['tph'] < 1] = -1
    
    sag['nivel_ton'][(sag['tph'] >= 1) & (sag['tph'] < 2000)] = 1
    sag['nivel_ton'][(sag['tph'] >= 2000) & (sag['tph'] < 3400)] = 2
    sag['nivel_ton'][(sag['tph'] >= 3400)] = 3
    
    sag_estacionario = sag[(sag['detenido'] == 0) & (sag['nivel_ton'] == 3)]
    
    # =============================================================================
    # Dinero
    # =============================================================================
    
    costos = {
        'usd/m3': 3,
        'usd/kwh': 0.15,
        'usd/ton': 2.34
        }
    
    sag_estacionario['agua_d'] = sag_estacionario['agua']*24
    sag_estacionario['agua_usd/d'] = sag_estacionario['agua_d']*costos['usd/m3']
    sag_estacionario['potencia_d'] = sag_estacionario['potencia']*24
    sag_estacionario['potencia_usd'] = sag_estacionario['potencia']*costos['usd/kwh']
    sag_estacionario['potencia_usd/d'] = sag_estacionario['potencia_usd']*24
    sag_estacionario['tpd'] = sag_estacionario['tph']*24
    sag_estacionario['usd/d'] = sag_estacionario['tpd']*costos['usd/ton']
    
    rep_dia = pd.DataFrame({
        'agua_usd/d': [sag_estacionario['agua_usd/d'].resample('D').mean().mean()],
        'potencia_usd/d': [sag_estacionario['potencia_usd/d'].resample('D').mean().mean()],
        'usd/d': [sag_estacionario['usd/d'].resample('D').mean().mean()]
        })
    
    rep_dia.to_excel('reportes/tasa_dolares.xlsx')
    
    print('Creación variables: agua_d, agua_usd/d, potencia_d, potencia_usd, \
    potencia_usd/d, tpd, usd/d')
    print('Reporte tasa dolares diarias de agua, potencia y tph en "reportes/tasa_dolares.xlsx"')

    return sag_estacionario


def reporte_minmax(sag, seed=2019, variables=['velocidad', 'tph']):
    
    """Función que estima la velocidad y el tratamiento de mineral mínimo"
    
    Parameters
    -------------
    
    sag: data sag estructurada albergado en 'datos/sag.pkl'.
    
    seed: semilla para los algoritmos kmeans y dbscan.
    
    variables: variables a estimar. En este caso solo permite la velocidad
    y el tph de tratamiento.

    Returns: 
        
        Reporte en consola.
    -------
    """
    
    sag = pd.read_pickle('../../datos/sag.pkl')
    
    # muestreo
    np.random.seed(seed)
    n_rand = np.random.randint(0, len(sag), 1000)
    pd_ = sag[variables].iloc[n_rand]
    
    # detectamos cuando el molino estuvo detenido (vel ~ 0)
    g_= grupos_densidad(pd_['velocidad'], dbscan_parameters=[0.5, 5])
    pd_['g_vel'] = g_
    vel_min = pd_['velocidad'][pd_['g_vel']==1].max()
    print('vel min estimada:', vel_min)
    
    pd_['aux_tph'] = grupos_densidad(pd_['tph'],
                                     kmean_parameters=[10, 'k-means++', 100, 500],
                                     seed=seed)
    
    tph_min = desc_grupos(pd_, 'tph', 'aux_tph')[9].loc['min']
    print('tph min estimada:', tph_min)
    
    np.random.seed(seed)
    n_rand = np.random.randint(0, len(sag), 3000)
    pd_ = sag['tph'].iloc[n_rand]
    g_= grupos_densidad(pd_, dbscan_parameters=[100, 5], seed=seed)
    pd_ = pd.DataFrame(pd_)
    pd_['g_tph'] = g_
    tph_max = pd_['tph'][(pd_['g_tph'] == -1) & (pd_['tph'] > 8000)].min()
    print('tph max estimada:', tph_max)
