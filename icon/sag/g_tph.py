import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wa
# from funciones.stats_code import (desc_grupos, descriptivo)
# from funciones.summary import (summary)
# from funciones.f_series import (mean_between)
# from sklearn.cluster import (KMeans, DBSCAN)
# wa.filterwarnings('ignore')
# sns.set('paper')
# pd.set_option('max.columns', 100)

# import data
# sag_estacionario = pd.read_pickle('../../../../datos/sag_estacionario1.pkl')

def g_tph(sag_estacionario):
    
    """Función que genera los clusters basados en el tratamiento de mineral
    junto a su análisis estadistico. Estos son agregados en el grupos 2 y grupos 3 del df.
    
    * grupos 3 corresponde a los mismos clusters del grupo 2, solo que contiene
      subclusters para los grupos con mayor tratamiento (6, 7, 8 y 9).
    
    Parameters
    -------------
    
    sag_estacionario: data sag estructurada albergada en '../../../datos/sag_estacionario1.pkl'.

    Returns: 
        
        ADS en formato dataframe (Pandas).
        Graficos y estadisticas
    -------
    """
    # =============================================================================
    # Kmeans
    # =============================================================================
    pd_ = sag_estacionario.copy()
    y = 'e_especifica_m'
    
    pd_['grupos'] = -1
    
    np.random.seed(2019)
    X = pd_[['tph', y]].values
    clusters = KMeans(30)
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    print('1er cluster (kmeans30): grupos categorias 0, 1, ...,  29, 30')
    
    # graficos exploratorios
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
                    data=pd_, hue='groups', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y energía específica (Kwh/T)\n\
              Kmeans30*')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='potencia', y=y,
                    data=pd_, hue='groups', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y potencia (Kwh)\nKmeans30*')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='velocidad', y=y,
                    data=pd_, hue='groups', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y velocidad (rpm)\nKmeans30*')
    plt.show()
    
    pd_['grupos'] = pd_['groups']
    
    # =============================================================================
    # Recategorizaciones
    # =============================================================================
    
    # grupos 7 == 9
    pd_['grupos'][pd_['grupos'] == 7] = 9
    print(' Recategorizacion grupo 7 == 9')
    
    # recategorizacion: toleramos el kmeans hasta el grupo 5200
    print(' categorias mayores: >5641 (codificados como -1)')
    pd_['grupos'][pd_['tph'] > 5641] = -1
    
    # graficos exploratorios
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
                    data=pd_[pd_['grupos'] == -1], hue='groups', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y energía específica (Kwh/T)\n\
              Kmeans30 (grupos mayores)*')
    plt.show()
    
    # reduccion de grupos: realizamos kmeans al grupo -1 (grupos mayores)
    np.random.seed(2019)
    X = pd_[pd_['grupos'] == -1][['tph', y]].values
    clusters = KMeans(4)
    groups = clusters.fit(X)
    pd_['grupos'][pd_['grupos'] == -1] = groups.labels_ + 50
    print(' Recategorización grupos mayores -1 (kmeans4): categorias 50, 51, 52, 53')
    
    # graficos exploratorios
    y = 'e_especifica'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
                    data=pd_, hue='grupos', s=5, palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y energía específica (Kwh/T)')
    
    # =============================================================================
    # Analisis de las caract de mineral clusters 1, 2, 3, 4, 5, 6, 7, 8 y 9
    # =============================================================================
    
    # grupos aceptables TPH: 51, 28, 9
    # grupos bajos: 3, 21, 13
    # grupos altos: 53, 50, 52
    
    # orden de los grupos
    pd_['grupos'] = pd_['grupos'] + 100
    pd_['grupos'][pd_['grupos'] == 121] = 1
    pd_['grupos'][pd_['grupos'] == 103] = 2
    pd_['grupos'][pd_['grupos'] == 113] = 3
    pd_['grupos'][pd_['grupos'] == 128] = 4
    pd_['grupos'][pd_['grupos'] == 109] = 5
    pd_['grupos'][pd_['grupos'] == 151] = 6
    pd_['grupos'][pd_['grupos'] == 153] = 7
    pd_['grupos'][pd_['grupos'] == 150] = 8
    pd_['grupos'][pd_['grupos'] == 152] = 9
    print(' Recategorización grupos menores: 1, 2, .., 9')
    
    print(' Agregadas categorias a sag_estacionario')
    sag_estacionario = pd_.copy()
    desc = desc_grupos(pd_, 'tph', 'grupos')
    
    ## en pd_ se agregaron los spi y luego se agrgaron a sag_estacioanrio
    
    # agregamos info del spi
    print()
    print('Agregando spi y bolas a sag_estacionario')
    sag_diario = pd.read_pickle('../../../../datos/sag_l2.pkl')
    bolas = pd.read_pickle('../../../../datos/bolas.pkl')
    sag_diario = sag_diario[['bwi_ls_l2', 'spi_ls_l2']]
    sag_diario = pd.concat([sag_diario, bolas['consumo_acero']], axis=1)
    
    pd_ = pd.concat([pd_, sag_diario.resample('H').ffill()], axis=1)
    pd_.reset_index(inplace=True)
    pd_.sort_values('index', inplace=True)
    pd_.set_index('index', inplace=True)
    
    pd_['spi'] = pd_['spi_ls_l2'].ffill()
    pd_['bwi'] = pd_['bwi_ls_l2'].ffill()
    pd_['consumo_acero'] = pd_['consumo_acero'].ffill()
    del pd_['spi_ls_l2']
    del pd_['bwi_ls_l2']
    del bolas
    del sag_diario
    pd_.dropna(inplace=True)
    
    print(' * detección outlier: reemplazamos por la media movil q=1')
    # outlier!
    pd_['bwi'][pd_['bwi'] < 8] = np.nan
    pd_ = mean_between(pd_)
    
    print('')
    print(' graficos decriotivos (boxplot)')
    # replace
    sag_estacionario = pd_.copy()
    
    # boxplot
    sns.boxplot(x='grupos', y='spi', data=pd_)
    plt.title('spi grupos')
    plt.show()
    sns.boxplot(x='grupos', y='bwi', data=pd_)
    plt.title('bwi grupos')
    plt.show()
    sns.boxplot(x='grupos', y='gruesos', data=pd_)
    plt.title('gruesos grupos')
    plt.show()
    sns.boxplot(x='grupos', y='finos', data=pd_)
    plt.title('finos grupos')
    plt.show()
    sns.boxplot(x='grupos', y='medios', data=pd_)
    plt.title('medios grupos')
    plt.show()
    sns.boxplot(x='grupos', y='tph', data=pd_)
    plt.title('tph grupos')
    plt.show()
    sns.boxplot(x='grupos', y='agua', data=pd_)
    plt.title('agua grupos')
    plt.show()
    
    print(' graficos decriotivos (barras)')
    # histogramas
    for i in np.sort(pd_['grupos'].unique()):
    
        idx_aux = pd.Series(summary(pd_['spi'][pd_['grupos'] == i],
                                    NAN=False).index)
        idx_aux = round(idx_aux, 1)
        idx_aux = idx_aux.astype(str)
        
        h_aux = summary(pd_['spi'][pd_['grupos'] == i], NAN=False) \
                        ['Frecuencia']/desc[i].loc['count']
    
        plt.barh(idx_aux.iloc[0:5], h_aux.iloc[0:5], color='#f86a25')
        plt.title('spi g'+str(i))
        plt.show()
        
    print(' graficos decriotivos (histogramas)')
    for i in np.sort(pd_['grupos'].unique()):
        sns.distplot(pd_['bwi'][pd_['grupos'] == i])
        plt.title('bwi g '+str(i))
        plt.show()
    
    for i in np.sort(pd_['grupos'].unique()):
        sns.distplot(pd_['gruesos'][pd_['grupos'] == i])
        plt.title('gruesos g '+str(i))
        plt.show()
    
    for i in np.sort(pd_['grupos'].unique()):
        sns.distplot(pd_['medios'][pd_['grupos'] == i])
        plt.title('medios g '+str(i))
        plt.show()
    
    for i in np.sort(pd_['grupos'].unique()):
        sns.distplot(pd_['finos'][pd_['grupos'] == i])
        plt.title('finos g '+str(i))
        plt.show()
    
    print()
    print('2do cluster:')
    # =============================================================================
    # Clustres grupos 1, 2 y 3
    # =============================================================================
    
    # realizamos clusters considerando spi para los grupos 1, 2 y 3 ya que
    # son los que presentan mayor diferencia
    print( ' Grupos 1, 2 y 3 (Kmeans): sumado 100, 200 o 300 dependiendo')
    # grupo de simulacion / grupo concentración especifica tph
    sag_estacionario['grupos 2'] = np.nan
    sag_estacionario['grupos 3'] = np.nan
    
    # grupo 1
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 1)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['spi', 'aux']].values
    clusters = KMeans(5, init='random')
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 100
    
    desc = desc_grupos(pd_, 'spi', 'groups')
    
    y = 'tph'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='spi', y=y, data=pd_, hue='groups', s=50, palette='dark')
    plt.title('Relación ingreso mineraly SPI (grupo 1)')
    plt.show()
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 1)] = pd_['groups']
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 1)] = pd_['groups']
    
    # grupo 2
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 2)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['spi', 'medios']].values  ## pues se aprecian diferencias entre con medios
    clusters = KMeans(2, init='random')
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 200
    
    desc = desc_grupos(pd_, 'medios', 'groups')
    
    y = 'tph'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='spi', y=y, data=pd_, hue='groups', s=50, palette='dark')
    plt.title('Relación ingreso mineraly SPI (grupo 2)')
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 2)] = pd_['groups']
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 2)] = pd_['groups']
    
    # grupo 3
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 3)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['spi', 'aux']].values
    clusters = KMeans(3, init='random')
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 300
    
    desc = desc_grupos(pd_, 'spi', 'groups')
    
    y = 'tph'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='spi', y=y, data=pd_, hue='groups', s=50, palette='dark')
    plt.title('Relación ingreso mineraly SPI (grupo 3)')
    plt.show()
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 3)] = pd_['groups']
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 3)] = pd_['groups']
    
    
    # grupo 4 (clusters descartado)
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 4)] = 4
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 4)] = 4
    
    # pd_ = sag_estacionario[(sag_estacionario['grupos']==4)]
    
    # from sklearn.cluster import KMeans
    # np.random.seed(2019)
    # pd_['aux'] = 1
    # X = pd_[['finos', 'aux']].values
    # clusters = KMeans(2, init='random')
    # groups = clusters.fit(X)
    # pd_['groups'] = groups.labels_
    
    # pd_['groups'] = pd_['groups'] + 400
    
    # desc = desc_grupos(pd_, 'tph', 'grupos')
    # sag_estacionario['grupos 2'][(sag_estacionario['grupos']==4)] = pd_['groups']
    
    # grupo 5 diferencia con medios y finos (clusters descartado)
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 5)] = 5
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 5)] = 5
    
    # pd_ = sag_estacionario[(sag_estacionario['grupos']==5)]
    # from sklearn.cluster import KMeans
    # np.random.seed(2019)
    # pd_['aux'] = 1
    # X = pd_[['medios', 'finos']].values
    # clusters = KMeans(10, init='random')
    # groups = clusters.fit(X)
    # pd_['groups'] = groups.labels_
    
    # pd_['groups'] = pd_['groups'] + 500
    
    # desc = desc_grupos(pd_, 'tph', 'grupos')
    
    # y='tph'
    # plt.figure(figsize=(8,8))
    # sns.scatterplot(x='spi', y=y, data=pd_, hue='groups', s=50, palette='dark')
    # plt.title('Relación ingreso mineraly SPI (grupo 3)')
    
    # pd_['groups'] = pd_['groups'] + (5000 - 500)
    
    # ## join clusters
    # pd_['groups'][(pd_['groups']==5000) | (pd_['groups']==5001) |
    #               (pd_['groups']==5005) | (pd_['groups']==5006) |
    #               (pd_['groups']==5006) | (pd_['groups']==5007) |
    #               (pd_['groups']==5008)] = 500
    # pd_['groups'][pd_['groups'] != 500] = 501
    # sag_estacionario['grupos 2'][(sag_estacionario['grupos']==5)] = pd_['groups']
    
    # =============================================================================
    # Clusters por hacer
    # =============================================================================
    
    ## ================== 6, 7, 8, 9 ================================== ##
    print(' Grupos 6, 7, 8 y 9 (kmeans). sumado 600, 700, 800, 900 dependiendo')
    # * kmeans no logra captar los verdaderos clusters
    # usamos dbscan para identificar, identifica bien , pero aun asi deja datos
    # extremos
    
    # pd_ = sag_estacionario[(sag_estacionario['grupos']==6) |
    #                        (sag_estacionario['grupos']==7) |
    #                        (sag_estacionario['grupos']==8) |
    #                        (sag_estacionario['grupos']==9)]
    
    # 6 solo detectamos el grupo concentado en cierto tph
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 6)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['tph', 'e_especifica']].values
    clusters = DBSCAN(eps=0.5, min_samples=13)
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 600
    
    desc = desc_grupos(pd_, 'tph', 'groups')
    
    y = 'e_especifica'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
            data=pd_[(pd_['groups'] == 600) | (pd_['groups'] == 599)],
            hue='groups', s=25, palette='dark')
    plt.title('Grupo 6 (600: concentración detectada)')
    plt.show()
    
    # grupo concentrado en un valor en particular: 6000
    pd_['groups'][pd_['groups'] != 600] = 6
    pd_['groups'][pd_['groups'] == 600] = 6000
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 6)] = 6
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 6)] = pd_['groups']
    
    
    # 7
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 7)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['tph', 'e_especifica']].values
    clusters = DBSCAN(eps=0.5, min_samples=13)
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 700
    
    desc = desc_grupos(pd_, 'tph', 'groups')
    
    # grupo concentrado en un valor en particular: 6000
    pd_['groups'][(pd_['groups'] == 704) | (pd_['groups'] == 734)] = 7000
    pd_['groups'][pd_['groups'] != 7000] = 7
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 7)] = 7
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 7)] = pd_['groups']
    
    # 8
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 8)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['tph', 'e_especifica']].values
    clusters = DBSCAN(eps=0.5, min_samples=21)
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 800
    
    desc = desc_grupos(pd_, 'tph', 'groups')
    
    
    # grupo concentrado en un valor en particular: 6000
    pd_['groups'][(pd_['groups'] == 805)] = 8000
    pd_['groups'][(pd_['groups'] == 806)] = 8001
    pd_['groups'][pd_['groups'] != 8000] = 8
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 8)] = 8
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 8)] = pd_['groups']
    
    # 9
    pd_ = sag_estacionario[(sag_estacionario['grupos'] == 9)]
    np.random.seed(2019)
    pd_['aux'] = 1
    X = pd_[['tph', 'e_especifica']].values
    clusters = DBSCAN(eps=0.5, min_samples=33)
    groups = clusters.fit(X)
    pd_['groups'] = groups.labels_
    
    pd_['groups'] = pd_['groups'] + 900
    
    desc = desc_grupos(pd_, 'tph', 'groups')
    
    y = 'e_especifica'
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y, data=pd_, hue='groups', s=25,
                    palette='dark', legend=True)
    plt.title('Grupo 9 (900: concentración detectada)')
    plt.show()
    
    # grupo concentrado en un valor en particular: 6000
    pd_['groups'][(pd_['groups'] == 900)] = 9000
    pd_['groups'][pd_['groups'] != 9000] = 9
    
    sag_estacionario['grupos 2'][(sag_estacionario['grupos'] == 9)] = 9
    sag_estacionario['grupos 3'][(sag_estacionario['grupos'] == 9)] = pd_['groups']
    
    desc = desc_grupos(pd_, 'tph', 'grupos 2')
    print('Grupos resultantes:')
    print('variable grupos 2: 100 101 102 103 104\n\
                       200 201 300 301 302\n\
                       300 301 302\n\
                       4 5 6 7 8 9')
    print()
    print('variable grupos 3: 100 101 102 103 104\n\
                       200 201 300 301 302\n\
                       300 301 302\n\
                       6 6000\n\
                       7 7000\n\
                       8 8000\n\
                       9 9000')
    
    # =============================================================================
    # Estadisticas de los grupos finales
    # =============================================================================
    # graficos exploratorios
    print('Graficos exploratorios finales')
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
                    data=sag_estacionario, hue='grupos 2', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y energía específica (Kwh/T)')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='potencia', y=y,
                    data=sag_estacionario, hue='grupos 2', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y potencia (Kwh)')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='velocidad', y=y,
                    data=sag_estacionario, hue='grupos 2', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y velocidad (rpm)')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='tph', y=y,
                    data=sag_estacionario, hue='grupos 3', s=5,
                    palette='dark', legend=True)
    plt.title('Relación ingreso mineral (TPH) y energía especifica (Kwh/T)')
    plt.show()
    
    # save
    # sag_estacionario.to_pickle('../../datos/sag_estacionario.pkl')
    # =============================================================================
    # Ajustes por efecto de modelo
    # =============================================================================
    # outliers
    print('Ajustes finales')
    # 6
    descriptivo(sag_estacionario[sag_estacionario['grupos 2'] == 6])
    sag_estacionario[(sag_estacionario['grupos 2'] == 6) &
                     (sag_estacionario['velocidad'] < 7)] = np.nan
    
    # 7
    sag_estacionario[(sag_estacionario['grupos 2'] == 7) &
                     (sag_estacionario['velocidad'] < 7)] = np.nan
    
    # 8
    sag_estacionario[(sag_estacionario['grupos 2'] == 8) &
                     (sag_estacionario['velocidad'] < 7)] = np.nan
    
    # 9
    sag_estacionario[(sag_estacionario['grupos 2'] == 9) &
                     (sag_estacionario['velocidad'] < 7)] = np.nan
    
    # 4
    sag_estacionario[(sag_estacionario['grupos 2'] == 4) &
                     (sag_estacionario['velocidad'] < 7)] = np.nan
    
    # 500
    sag_estacionario[(sag_estacionario['grupos 2'] == 500) &
                     (sag_estacionario['velocidad'] < 6)] = np.nan
    
    # 300 - 301
    sag_estacionario['grupos 2'][sag_estacionario['grupos 2'] == 301] = 300
    sag_estacionario['grupos 2'][sag_estacionario['grupos 2'] == 302] = 301
    # save nuevo sag_estacionario
    
    sag_estacionario['agua_especifica'] = sag_estacionario['agua']/ \
                                          sag_estacionario['tph']
    
    sag_estacionario = sag_estacionario.dropna()
    
    return sag_estacionario

