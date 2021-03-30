import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
# import warnings as w
# w.filterwarnings('ignore')
# pd.set_option('max.columns', 100)

def round5(x):
    """Función que redondea en miltiplos de 5".
    
    Parameters
    -------------
    
    x: numero e formato float.
        
    Returns: 
        
        numero float redondeado.
    -------
    """
    if x%1 >= 0.5:
        return round(x) + 0.5 - 1
    else:
        return round(x)

def g_nps(sag_estacionario):
    
    """Función que crea los grupos por nivel de stock pile. Se almacena en la
       variable "g_nps".
    
    Parameters
    -------------
    
    sag_estacionario:
        
    Returns: 
        
        ADS en formato dataframe (Pandas).
    -------
    """
    # grupos por nivel stok pile
    sag_estacionario['g_nps'] = 1
    sag_estacionario['g_nps'][(sag_estacionario['nivelpromedio_sp']>40) &\
                              (sag_estacionario['nivelpromedio_sp']<=60)] = 2
    sag_estacionario['g_nps'][sag_estacionario['nivelpromedio_sp']>60] = 3
    
    return sag_estacionario

    
def sag_estacionario_week(sag_estacionario):
    
    """Función que genera un reporte semanal promedio de las variables.
       Ademas de calcular la informacion del % potencia utilizado con respecto
       al maximo historico o el percentil 90.
       "
    
    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacioanria.
        
    Returns: 
        
        DataFrame con la informacion (Pandas).
    -------
    """
    sag_estacionario_w = sag_estacionario.resample('W').mean()
    sag_estacionario_w['spi_round'] = round(sag_estacionario_w['spi'])
    sag_estacionario_w = sag_estacionario_w.loc[:dt.datetime(2020, 3, 9)]
    sag_estacionario_w['%potenciamax'] = sag_estacionario_w['potencia']/24735
    sag_estacionario_w['%potencia90'] = sag_estacionario_w['potencia']/24006
    
    # informacion promedio de spi usados por semana

    # sag_estacionario_w.to_excel('reportes/sag_estacionario_porweek.xlsx')
    
    return sag_estacionario_w

def g_tph_week(sag_estacionario, sag_estacionario_w,
               clusters_tph=[[6055, 7905],
                             [5810, 8036, 7286],
                             [6960, 7561, 8082],
                             [8317, 7201, 6492],
                             [8019, 7497, 6994, 5081],
                             [5951, 6447, 6742, 7082],
                             [7867, 5949, 4876],
                             [4495, 5000, 5510, 5958, 6489, 6992],
                             [5747, 7814],
                             [7992, 5515, 7328, 4839, 8422, 6201],
                             [7983, 7477, 7049, 6486, 5890, 4848],
                             [3773, 5614, 8389],
                             [],
                             [5497, 5980, 6318, 6645, 7530, 8009],
                             [5012, 6204, 7008, 7588, 8123],
                             [6096, 7056, 7618, 8067],
                             [5525, 6002, 6554, 7567],
                             [5992, 6551, 7122, 7697, 8002, 8352],
                             [5996, 7467, 8052, 8570],
                             [6558, 7417, 8034],
                             [5995, 6501, 7470],
                             [6158],
                             [6511, 7481, 7767, 8159, 8485],
                             [6982, 7515, 7976],
                             [5097, 5940, 7165, 8058],
                             [],
                             [5864, 6462, 6951, 7328],
                             [6990, 7481, 7921]],
               clusters=[3, 3, 4, 5, 6, 6, 6, 9, 2, 6, 8, 6, 0, 8, 5, 5, 5, 7,
                         4, 6, 7, 5, 7, 4, 4, 0, 6, 3]):
    
    """Función que analiza las semanas y mantiene los clusters elegidos en.
       Agrega los resultados en la columna "grupos_situaciones".
       
       * En un principio se clusterizó mediante kmeans (seed 2019) y se eligieron 
        los clusters significativos a mano para obtener los promedios de los grupos donde
        reflejaban una concentración significativa del valor de tratamiento (tph).
        Estos se graficaron en esta función representadas con rectas -- en los graficos.
    
    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacioanria.
        
    sag_estacionario_w: df con la informacion por semana del molino sag en situacion estacionaria.
    
    clusters_tph: promedio de cluster que se eligieron para el análisis.
    
    clusters: id del cluster que se produce con el algoritmo KMeans de la libreria
              sklearn.cluster.
        
    Returns: 
        
        ADS en formato dataframe (Pandas).
    -------
    """

    sag_estacionario['spi_round'] = round(sag_estacionario['spi'])

    print('detección de grupos tph para cada semana')
    print('semana [clusters tph (promedios de grupos)]')

    sag_estacionario['grupos_situaciones'] = np.nan
    kmeans = {}

    for j in range(0, len(clusters_tph)):
        
        # indexamos el grupo indicado con unidad j00gtph.
        #                                        j -> numero de semana
        #                                        gtph -> grupo tph
        
        # a cada grupos de tph le llamaremos situacion
        
        aux = j*1000
        idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[j]))[0]
        
        # pd_ de la semana j
        pd_ = sag_estacionario.loc[
        dt.datetime(idx_date.year,
                    idx_date.month,
                    idx_date.day):idx_date+dt.timedelta(7)]
        pd_['grupos_situaciones'] = aux
        
        # redondeamos e_especifica a multiplos de 5
        pd_['e_especifica_round5'] = pd_['e_especifica'].apply(lambda x: round5(x))
        pd_['%potencia'] = pd_['potencia']/24006
        
        if clusters[j]==0:
            print(idx_date, '[]')
            sag_estacionario.loc[pd_.index, 'grupos_situaciones'] = pd_['grupos_situaciones']
            continue

        # clusters para identificar los grupos de tph para la semana j
        from sklearn.cluster import KMeans
        pd_['aux'] = 1
        X = pd_[['tph', 'aux']].values
        
        np.random.seed(2019)
        km = KMeans(clusters[j])
        km = km.fit(X)
        pd_['groups'] = km.labels_
        pd_['grupos_situaciones'] = pd_['grupos_situaciones'] + pd_['groups']

        # == graficos dispersiones == #
        sns.set('paper')
        y = 'tph'
        for v in ['presion']:
            plt.figure(figsize=(10,10))
            plt.plot(pd_[v], pd_[y], '.')
            plt.xlabel(v, size=20)
            plt.ylabel(y, size=20)
            for t in pd_.groupby('groups').mean()[y].unique():
                plt.axhline(y=t, linestyle='--',
                            color='black')
            plt.title('Dispersión semana: '+str(idx_date), size=15)
            plt.show()

        # == promedio de los grupos tph == #
        situaciones = pd_.groupby('groups').mean()[['tph', 'spi_round', '%potencia',
                                                    'bwi', 'nivelpromedio_sp', 'g_nps',
                                                    'agua', 'velocidad', 'porcSolido',
                                                    'gruesos', 'finos', 'medios',
                                      'potencia_usd', 'potencia_usd/d', 'agua_usd/d',
                                      'potencia', 'presion', 'e_especifica', 'pebbles',
                                      'impacto_critico', 'presion_optima', 'mpc_sag',
                                      'pot_esc', 'grupos_situaciones'
                                      ]]

        situaciones['tph'] = situaciones['tph'].astype(int)
        situaciones.set_index('tph', inplace=True)

        # agregamos el idx de los grupos de la semana correspondiente
        sag_estacionario.loc[pd_.index, 'grupos_situaciones'] = pd_['grupos_situaciones']
        
        # guardamos los df de las situaciones de tph en kmeans
        try:
            kmeans[idx_date] = situaciones.loc[clusters_tph[j]]
            print(idx_date, situaciones.index.tolist())
        except:
            print('Error',idx_date, situaciones.index.tolist(), j)
            
    # #save
    info_grupos = {}
    info_grupos['clusters'] = clusters
    info_grupos['clusters_tph'] = clusters_tph
    info_grupos['kmeans'] = kmeans
    # np.save('modelos/info_clusters_week-2.npy', info_grupos)
    
    # load
    # info_grupos = np.load('modelos/info_clusters_week-2.npy', allow_pickle=True).item()
    # clusters = info_grupos['clusters']
    # clusters_tph = info_grupos['clusters_tph']
    # kmeans = info_grupos['kmeans']
    # del info_grupos
            
    return (sag_estacionario, info_grupos)


def valores_optimos(sag_estacionario, kmeans,
                    min_tph=6476, max_tph=8022, cutoff_nps=40):
        
    """Función que retorna los valores optimos y simula valores iniciales para
       la plataforma"
    
    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacionaria y
                      columna "grupos_situaciones".
        
    kmeans: diccionario con los df de informacion promedio de las variales de
            cada semana.
        
    Returns: 
        
        ADS en formato dataframe (Pandas).
    -------
    """

    # =============================================================================
    # filtro de los datos
    # =============================================================================
    ## 6476 < tph < 8022 (filtro 80-20 tph)
    print()
    print('filtro 80-20 (6476 < tph < 8022)')
    print('semana [promedios de grupos]')

    # solo tomamos los clusters donde tengan datos dentro del 80-20
    kmeans2 = {}
    for date_idx in kmeans:
        kmeans2[date_idx] = kmeans[date_idx][(kmeans[date_idx].index>min_tph) &\
                                             (kmeans[date_idx].index<max_tph) &\
                                    (kmeans[date_idx]['nivelpromedio_sp']>cutoff_nps)]
        print(date_idx, kmeans2[date_idx].index.tolist())
    
    # obtenemos los datos
    idx_tph = []
    for date_idx in kmeans:
        idx_tph = idx_tph + kmeans2[date_idx]['grupos_situaciones'].values.tolist()

    pd_ = sag_estacionario[sag_estacionario['grupos_situaciones'].isin(idx_tph)]

    # estadisticas (medias) de las distintas situaciones (grupos_situaciones)
    grupos_est = pd_.groupby('grupos_situaciones').mean()[['tph', 'spi_round', 'nivelpromedio_sp',
                                                          'agua', 'porcSolido',
                                                          'finos', 'gruesos', 'medios', 
                                                          'velocidad',
                                                          'potencia', 'presion',
                                                          'pebbles', 'impacto_critico',
                                                          'e_especifica', 'consumo_acero'
                                                          ]]
    grupos_est['spi_round'] = grupos_est['spi_round'].astype(int)
    
    # spi_round es el promedio de los spi de cada grupo detectado semanalmente

    grupos_est.sort_values(['spi_round', 'tph'], inplace=True)
    grupos_est.reset_index(inplace=True)
    
    grupos_est.to_excel('reportes/sag_estacionario_porweek_tph8020.xlsx')
    print('reporte medias por situaciones de tph exportado en reportes/sag_estacionario_porweek_tph8020.xlsx')

    # ==* valores optimos *== #
    # seran los casos donde para cada spi el procesamiento (TPH) fue máximo

    # grupos optimos
    optimos = pd.DataFrame()
    idx_max_tph = grupos_est.groupby('spi_round').max()['tph']
    for i in range(len(idx_max_tph)):
        optimos = pd.concat([optimos, 
                             grupos_est[grupos_est['tph']==idx_max_tph.iloc[i]]])

    optimos = optimos[['tph', 'spi_round', 'agua', 'porcSolido', 'finos', 'gruesos',
                      'medios', 'velocidad', 'potencia', 'presion', 'pebbles',
                      'e_especifica', 'consumo_acero']]
    optimos.set_index('spi_round', inplace=True)
    optimos = optimos.transpose()
    optimos = optimos.to_dict()

    # simulamos inicilaes aleatorios para el front
    print('simulación iniciales aleatorios para el front')
    mins_ = grupos_est.groupby('spi_round').min()
    maxs_ = grupos_est.groupby('spi_round').max()
    
    inicial = {}
    for i in range(38, 53):
        inicial[i] = {}
        for v in ['tph', 'finos', 'gruesos', 'medios', 'agua', 'velocidad',
                  'presion', 'potencia', 'pebbles', 'e_especifica']:
            if mins_[v].loc[i] == maxs_[v].loc[i]:
                inicial[i][v] = np.random.normal(mins_[v].loc[i], 1)
            else:
                inicial[i][v] = np.random.uniform(mins_[v].loc[i],
                                           maxs_[v].loc[i], 1)[0]

    # inicial y optimo
    inicial_optimo = {'inicial': inicial,
                      'optimo': optimos}

    # # save inicial y optimo
    # np.save('parametros/inicial_optimo.npy', inicial_optimo)
    # print('inicial y optimo guardado en "parametros/inicial_optimo.npy"')
    
    # agregamos los spi idx a los datos sag_estacionario
    print('agregamos los spi index a los datos de sag_estacionario: categoria spi_round_idx')
    sag_estacionario['spi_round_idx'] = np.nan
    for i in range(len(grupos_est)):
        sag_estacionario['spi_round_idx'][
        sag_estacionario['grupos_situaciones']==\
            grupos_est.loc[i, 'grupos_situaciones']] = grupos_est.loc[i, 'spi_round']
    
    return (sag_estacionario, inicial_optimo, grupos_est)

# # save parametros
# np.save('parametros/inicial_optimo.npy', inicial_optimo)



