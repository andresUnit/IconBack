import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


### llena los missing values con el promedio entre los valores 
### si cero es True, significa que rellenará los ceros y los Nan
def mean_between(serie, cero = False):
    if cero == True:
        serie=serie.replace(0, np.nan)
    serie_bfill=serie.fillna(method='bfill')
    serie_ffill=serie.fillna(method='ffill')
    
    serie=(serie_bfill+serie_ffill)/2
    return serie

## plotea los resultados de un seasonal_descompose de la libreria
## statsmodels
def plot_decompose(DecomposeResult, figsize=(15,13), res_test_stationary='adfuller',
                   title = ''):
    
    # DecomposeResult=decompose_additive
    
    
    ### VER COMO PLOTEAR LOS STRING CON XTICKS 
    # plt.plot(np.arange(0,len(DecomposeResult.observed)), DecomposeResult.observed, linewidth=0.8)
    
    # x=np.arange(0,len(DecomposeResult.observed))*3
    # intersection(x.tolist(), np.arange(0,len(DecomposeResult.observed)).tolist())
    # plt.xticks([0,1,4,8], labels = DecomposeResult.observed.index)
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    ax1.plot(DecomposeResult.observed.index, DecomposeResult.observed, linewidth=0.8)
    ax1.plot(DecomposeResult.trend.index, DecomposeResult.trend, linestyle='--')

    ax2.plot(DecomposeResult.seasonal.index, DecomposeResult.seasonal)

    ax3.plot(DecomposeResult.resid.index, DecomposeResult.resid, color='black',
             linewidth=0.8)

    #plt.show()
    
    aux_info = []
    ## print information serie
    from statsmodels.tsa.stattools import adfuller
    results_obs = adfuller(DecomposeResult.observed.dropna()) # tupla
    print('values observed')
    print('Adfully test')
    print('p-value:', results_obs[1])
    if results_obs[1] < 0.01:
        print('1% : Stationary')
    if results_obs[1] < 0.05:
        print('5% : Stationary')
    if results_obs[1] < 0.10:
        print('10% : Stationary')
    
    print('stats')
    for key in results_obs[4]:
        print(key,':',results_obs[4][key])
    
    print()

    information = pd.DataFrame(['p-value', 'Stat 1%', 'Stat 5%', 'Stat 10%'])
    information.set_index(0, inplace = True)
    information['test'] = np.repeat('Adfully test', len(information))
    information.reset_index().set_index([0, 'test'])
    information['values observed'] = [results_obs[1], results_obs[4]['1%'],
                                     results_obs[4]['5%'], results_obs[4]['10%']]
    
    ## print information residuals
    from statsmodels.tsa.stattools import adfuller
    results_res = adfuller(DecomposeResult.resid.dropna())
    print('residuals results')
    print('Adfully test')
    print('p-value:', results_res[1])
    if results_res[1] < 0.01:
        print('1% : Stationary')
    if results_res[1] < 0.05:
        print('5% : Stationary')
    if results_res[1] < 0.10:
        print('10% : Stationary')
    
    print('stats')
    for key in results_res[4]:
        print(key,':',results_res[4][key])
        

    information['residuals'] = [results_res[1], results_res[4]['1%'],
                                     results_res[4]['5%'], results_res[4]['10%']]
    
    return information
    
def plot_additivemodel(pd_serie, freq = None):
    
    # pd_serie = pd_[var]
    # freq = 16
    
    if pd_serie.isnull().sum() > 0:
        pd_serie = pd_serie.dropna()
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    decompose_additive = seasonal_decompose(pd_serie, freq = freq)
    info = plot_decompose(decompose_additive)
    
    return info
    
        
## debe ingresarse los datos con index date
def Graph_historial(antes_serie, post_serie=None, title=None, type_='mean'):
    
    ## cant/distancia 
    plt.figure(figsize=(15,3))
    plt.plot(antes_serie.resample('D').mean().index,
             antes_serie.resample('D').mean(), alpha=0.5)
    plt.plot(antes_serie.resample('W').mean().index,
             antes_serie.resample('W').mean(), alpha=0.8)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel('promedio: '+ str(antes_serie.resample('D').mean().mean()))
    plt.legend(['diario', 'semanal'])
    plt.show()
    ## post
    if post_serie!=None:
        plt.figure(figsize=(15,3))
        plt.plot(post_serie.resample('D').mean().index,
                 post_serie.resample('D').mean(), alpha=0.5)
        plt.plot(post_serie.resample('W').mean().index,
                 post_serie.resample('W').mean(), alpha=0.8)
        plt.xticks(rotation=45)
        plt.title(title)
        plt.xlabel('promedio: '+ str(post_serie.resample('D').mean().mean()))
        plt.legend(['diario', 'semanal'])
        plt.show()
        
def ACF_PACF(pd_serie, test_stationary='adfuller', alpha=0.05, lags=50, figsize=(5,5),
        print_informationIC=False):

    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.stattools import pacf
    serie_acf=acf(pd_serie)
    serie_pacf=pacf(pd_serie)
    
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(pd_serie, lags=lags, alpha=alpha)
    plt.show()
    
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(pd_serie, lags=lags, alpha=alpha)
    plt.show()
    
    ## information confidance bound
    if print_informationIC:
        from PIL import Image
        img = Image.open('imagenes/IC_information.png')
        img.show()
        del img
        
    ## test stationary
    print('Stationary Test')
    print('===============')
    if test_stationary == 'adfuller':
        from statsmodels.tsa.stattools import adfuller
        results_adfuller = adfuller(pd_serie)
        print('Adfully test')
        print('p-value:', results_adfuller[1])
        if results_adfuller[1] < 0.01:
            print('1% : Stationary')
        if results_adfuller[1] < 0.05:
            print('5% : Stationary')
        if results_adfuller[1] < 0.10:
            print('10% : Stationary')
        
        print('stats')
        for key in results_adfuller[4]:
            print(key,':',results_adfuller[4][key])
    
    if test_stationary == 'kpss':
        from statsmodels.tsa.stattools import kpss
        results_kpss = kpss(pd_serie)
        print('KPSS results')
        print('p-value:', results_kpss[1])
        print('lags:',results_kpss[2])
        
        for key in results_kpss[3]:
            print(key,':',results_kpss[3][key])
        print()
    
    return (serie_acf, serie_pacf)

def resampleWeek(pd_serie, function='mean'):
    
    if function=='mean':
        week = pd.DataFrame(pd_serie.resample('W').mean())
    if function=='sum':
        week = pd.DataFrame(pd_serie.resample('W').sum())
    if function=='max':
        week = pd.DataFrame(pd_serie.resample('W').max())
    if function=='min':
        week = pd.DataFrame(pd_serie.resample('W').min())
        
    week['week'] = week.index.week
    week.set_index('week', inplace=True)
    return week


def grupos_densidad(x, dbscan_parameters = None, kmean_parameters = None,
                    plt_figure = True, reporte_path = None,
                    boxplot_extremos = False, seed = None):
    
    # x = df1['%Cu Conc final']
    # kmean_parameters = [7, 'k-means++', 100, 500]
    # seed = np.random.seed(2019)

    if seed != None:
        np.random.seed(seed)

    var_name = x.name

    x = pd.DataFrame(x.dropna())
    x['aux'] = 0
    
    if dbscan_parameters == None and kmean_parameters == None:
        raise TypeError("indique parametros para 'dbscan_parameters' o 'kmean_parameters'")
    
    if dbscan_parameters != None and kmean_parameters != None:
        raise TypeError('No puede usar kmeans y dbscan al mismo tiempo')
    
    if dbscan_parameters != None:
        from sklearn.cluster import DBSCAN
        g_db = DBSCAN(eps = dbscan_parameters[0],
                      min_samples = dbscan_parameters[1])
        g_db = g_db.fit(x.values)
        x['g_'] = g_db.labels_
        del g_db
        del DBSCAN

    if kmean_parameters != None:
        from sklearn.cluster import KMeans
        g_km = KMeans(n_clusters = kmean_parameters[0],
                      init = kmean_parameters[1],
                      n_init = kmean_parameters[2],
                      max_iter = kmean_parameters[3])
        g_km = g_km.fit(x.values)
        x['g_'] = g_km.labels_
        
        # ordenamos las categorias de menor a mayor
        v_aux = x.groupby('g_').min().reset_index()
        v_aux.sort_values(var_name, inplace=True)
        v_aux['g_new'] = np.arange(0,len(v_aux))
        x['g_new'] = np. nan
        for i in range(len(v_aux['g_'])):
            x['g_new'][x['g_']==i] = v_aux['g_new'][v_aux['g_']==i].values[0]
        x['g_'] = x['g_new'].astype(int)
        
        del x['g_new']
        del v_aux
        del g_km
        del KMeans
        
        # ordenamos las variables
        

    ## medidas estadisticas
    from funciones.stats_code import descriptivo
    desc = x[[var_name, 'g_']].groupby('g_').apply(lambda x: descriptivo(x)).reset_index()
    desc = desc[desc['index']!='g_']
    del desc['index']
    desc.set_index('g_', inplace=True)
    
    ## boxplot
    from seaborn import boxplot
    plt.figure(figsize = (10,10))
    boxplot(x = 'g_',
            y = var_name, data = x, color = 'white')
    plt.title(var_name)
    plt.xlabel('grupos')
    plt.ylabel(var_name)
    
    x_g = x['g_'].unique().tolist()
    
    ## plot
    if plt_figure == True:
        plt.figure(figsize = (15, 3))
    plt.plot(x[var_name])
    for g in x_g:
        plt.plot(x[var_name][x['g_'] == g], '.', alpha = 0.7)
    plt.legend(['datos'] + x_g)
    plt.title(var_name)
    
    return x['g_'].values, desc
