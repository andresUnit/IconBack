import pandas as pd
import datetime as dt
import numpy as np
import warnings as wr
import seaborn as sns
import matplotlib.pyplot as plt
# from g_weekSpiTphNSP import round5
# wr.filterwarnings('ignore')
# pd.set_option('max.columns', 100)

# sag_estacionario = pd.read_pickle('../../../../datos/sag_estacionario.pkl')
# sag_estacionario['spi_round'] = round(sag_estacionario['spi'])


def estudio_cluster_week(sag_estacionario, sag_estacionario_w):
    
    """Función que se utilizó para estudiar para cada semana el patron
       "
    
    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacioanria.
        
    Returns: 
        
        No retorna nada, sólo se uso y los resultados estan al final en el
        apartado "Elecciones de cluster kmean para cada semana".
    -------
    """
    # =============================================================================
    # por semana (clusters a mano)
    # vamos cambiando por semana (j) y vamos analizando los clustrs
    # =============================================================================
    
    # *** No es automatico ***
    
    j = 0 # numero de semana
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[j]))[0]
    pd_ = sag_estacionario.loc[
    dt.datetime(idx_date.year,
                idx_date.month,
                idx_date.day):idx_date+dt.timedelta(7)]

    pd_['e_especifica_round5'] = pd_['e_especifica'].apply(lambda x: round5(x))
    pd_['%potencia'] = pd_['potencia']/24006
    
    #kmeans
    from sklearn.cluster import KMeans
    pd_['aux'] = 1
    X = pd_[['tph', 'aux']].values
    
    np.random.seed(2019)
    km = KMeans(3) # elegimos los clusters
    km = km.fit(X)
    pd_['groups'] = km.labels_
    
    # == dispersiones == #
    sns.set('paper')
    y = 'tph'
    for v in ['presion']:
        plt.figure(figsize=(10,10))
        plt.plot(pd_[v], pd_[y], '.')
        plt.xlabel(v)
        plt.ylabel(y)
        for t in pd_.groupby('groups').mean()[y].unique():
            plt.axhline(y=t, linestyle='--',
                        color='black')
        plt.title('Dispersión semana: '+str(idx_date))
        plt.show()
    
    situaciones = pd_.groupby('groups').mean()[['tph', 'spi_round', '%potencia',
                                                'bwi', 'nivelpromedio_sp',
                                                'agua', 'velocidad', 'porcSolido',
                                                'gruesos', 'finos', 'medios',
                                  'potencia_usd', 'potencia_usd/d', 'agua_usd/d',
                                  'potencia', 'presion', 'e_especifica', 'pebbles',
                                  'impacto_critico', 
                                  ]]
    
    # dejamos como idx de cada cluster el tph
    situaciones['tph'] = situaciones['tph'].astype(int)
    situaciones.set_index('tph', inplace=True)
    print(situaciones.index)
    
    # =========================================================================
    # Eleccion de cluster kmeans para cada semana
    # =========================================================================
    
    kmeans = {}
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[0]))[0]
    kmeans[idx_date] = situaciones.loc[[6055, 7905]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[1]))[0]
    kmeans[idx_date] = situaciones.loc[[5810, 8036, 7286]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[2]))[0]
    kmeans[idx_date] = situaciones.loc[[6960, 7561, 8082]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[3]))[0] 
    kmeans[idx_date] = situaciones.loc[[8317, 7201, 6492]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[4]))[0]
    kmeans[idx_date] = situaciones.loc[[8019, 7497, 6994, 5081]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[5]))[0]
    kmeans[idx_date] = situaciones.loc[[5951, 6447, 6742, 7082]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[6]))[0]
    kmeans[idx_date] = situaciones.loc[[7867, 5949, 4876]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[7]))[0]
    kmeans[idx_date] = situaciones.loc[[4495, 5000, 5510, 5958, 6489, 6992]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[8]))[0]
    kmeans[idx_date] = situaciones.loc[[5747, 7814]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[9]))[0]
    kmeans[idx_date] = situaciones.loc[[7992, 5515, 7328, 4839, 8422, 6201]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[10]))[0]
    kmeans[idx_date] = situaciones.loc[[7983, 7477, 7049, 6486, 5890, 4848]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[11]))[0]
    kmeans[idx_date] = situaciones.loc[[3773, 5614, 8389]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[12]))[0]
    kmeans[idx_date] = situaciones.loc[[]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[13]))[0]
    kmeans[idx_date] = situaciones.loc[[5497, 5980, 6318, 6645, 7530, 8009]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[14]))[0]
    kmeans[idx_date] = situaciones.loc[[5012, 6204, 7008, 7588, 8123]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[15]))[0]
    kmeans[idx_date] = situaciones.loc[[6096, 7056, 7618, 8067]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[16]))[0]
    kmeans[idx_date] = situaciones.loc[[5525, 6002, 6554, 7567]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[17]))[0]
    kmeans[idx_date] = situaciones.loc[[5992, 6551, 7122, 7697, 8002, 8352]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[18]))[0]
    kmeans[idx_date] = situaciones.loc[[5996, 7467, 8052, 8570]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[19]))[0]
    kmeans[idx_date] = situaciones.loc[[6558, 7417, 8034]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[20]))[0]
    kmeans[idx_date] = situaciones.loc[[5995, 6501, 7470]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[21]))[0]
    kmeans[idx_date] = situaciones.loc[[6158]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[22]))[0]
    kmeans[idx_date] = situaciones.loc[[6511, 7481, 7767, 8159, 8485]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[23]))[0]
    kmeans[idx_date] = situaciones.loc[[6982, 7515, 7976]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[24]))[0]
    kmeans[idx_date] = situaciones.loc[[5097, 5940, 7165, 8058]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[25]))[0]
    kmeans[idx_date] = situaciones.loc[[]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[26]))[0]
    kmeans[idx_date] = situaciones.loc[[5864, 6462, 6951, 7328]]
    
    idx_date = list(pd.DataFrame(sag_estacionario_w.iloc[27]))[0]
    kmeans[idx_date] = situaciones.loc[[6990, 7481, 7921]]
    
    # cluster significativos elegidos
    clusters = [3, 3, 4, 5, 6, 6, 6, 9, 2, 6, 8, 6, 0, 8, 5, 5, 5, 7, 4, 6, 7, 5,
                7, 4, 4, 0, 6, 3]
    
    clusters_tph = [[6055, 7905],
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
    [6990, 7481, 7921]]
    
    save_info = {}
    save_info['clusters'] = clusters
    save_info['clusters_tph'] = clusters_tph
    save_info['kmeans'] = kmeans

# save dict
# np.save('modelos/info_clusters_week-2.npy', save_info)

