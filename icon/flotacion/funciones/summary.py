import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

def summary(pd_serie, category = True, percentiles = None, 
               include = None, exclude = None, NAN = True, graph=False):
    
    nan_aux = int(pd_serie.isnull().sum())  ### suma de los missing values

    if(category == True):
        count_categorias = pd_serie.value_counts()  ### frecuancia de categorias
        if( NAN == True):
            count_categorias.loc['Nan'] = nan_aux ### agregar Nan
        #### ver como ponerle nombre 'Frecuencia' a la fila del df
        count_categorias=pd.DataFrame(count_categorias)
        count_categorias.reset_index(inplace=True)
        count_categorias['Frecuencia']=count_categorias.iloc[:,1]
        count_categorias['Categoria']=count_categorias.iloc[:,0]
        del count_categorias[list(count_categorias)[0]]
        del count_categorias[list(count_categorias)[0]]
        count_categorias.set_index(['Categoria'], inplace=True)
        
        if graph==True:
            plt.bar(count_categorias.index, count_categorias['Frecuencia'])
        return(count_categorias)
    else:
        count_categorias = pd_serie.describe(percentiles = percentiles,
                         include = include, exclude = exclude)  ### frecuancia de categorias
        if(NAN == True):
            count_categorias.loc['Nan'] = nan_aux ### agregar Nan

        if graph==True:
            print('Grafico solo para variables categoricas')
            raise ValueError
                
        return(count_categorias)
        

def freq(df, name, name_col='freq'):
    if type(name)==str:
        name=[name]
    df[name_col]=1
    return(df.groupby(name).count()[[name_col]])

## saca cualquier digito de un str
def sacar_digitos(str_):
    str_=''.join([i for i in str_ if not i.isdigit()])
    return str_

## saca los ultimos numeros de un str
def sacar_ultimosNum(str_):
    aux=str_[-1]
    k='not_finished'
    while k=='not_finished':
        print('while_yet')
        try:
            aux=str_[-1]
            aux=int(aux)
            str_=str_[:-1]
        except ValueError:
            k='finished'
    return str_

# Separa un df segun cierta categoria poniendo los nuevos dfs en una lista
def SubSet(df, var_name):
    dfs=[]
    for category in df[var_name].unique():
        print(category)
        eq_=df[df[var_name]==category]
        dfs.append(eq_)

# retorna en un str la fecha indicada de una semana y año en particular
def get_week_day(week_year, inicio = 2014, fin = 2023):
     #week_year = (52, 2017)

     #week_year = (hh_contratista['semana'].iloc[i],hh_contratista['año'].iloc[i])

    dates = np.arange(dt.datetime(inicio, 1, 1), dt.datetime(fin, 1, 1), dt.timedelta(days = 1))
    dates = pd.DataFrame({'dates': dates})
    dates['week'] = dates['dates'].dt.week
    dates.set_index('dates', inplace = True)
    dates_week = dates.resample('W').mean()
    dates_week.reset_index(inplace = True)
    dates_week['day'] = dates_week['dates'].dt.day
    dates_week['month'] = dates_week['dates'].dt.month
    dates_week['year'] = dates_week['dates'].dt.year
    dates_week.reset_index(inplace = True)
    week_date = dates_week[(dates_week['week'] == week_year[0]) & \
                           (dates_week['year'] == week_year[1])]
    if len(week_date) == 0:
        print('No existe dicha semana, verifica si la semana 52 paso a ser \
              del año siguiente o existe una semana 53 del año siguiente')
        return ''
    return week_date['dates'].astype('str').iloc[-1]

## error cuadratico medio
def mse(pd_serie1, pd_serie2):
    #pd_serie1 = pred_
    #pd_serie2 = test
    pd_serie1 = pd_serie1.copy()
    pd_serie2 = pd_serie2.copy()
    df_1 = pd.DataFrame(pd_serie1)
    df_2 = pd.DataFrame(pd_serie2)
    
    df_1.reset_index(drop = True, inplace=True)
    df_2.reset_index(drop = True, inplace=True)

    pd_serie1 = df_1.iloc[:,0]
    pd_serie2 = df_2.iloc[:,0]
    
    mse = pd_serie1 - pd_serie2
    
    mse = mse.apply(lambda x : x**2).mean()
    return mse

## error absoluto medio con su desviacion estandar
def mae(pd_serie1, pd_serie2):
    
    # pd_serie1 = test['potencia'].iloc[1:]
    # pd_serie2 = test['prediccion'].iloc[1:]

    pd_serie1 = pd_serie1.copy()
    pd_serie2 = pd_serie2.copy()
    
    df_1 = pd.DataFrame(pd_serie1)
    df_2 = pd.DataFrame(pd_serie2)
    
    df_1.reset_index(drop = True, inplace=True)
    df_2.reset_index(drop = True, inplace=True)

    pd_serie1 = df_1.iloc[:,0]
    pd_serie2 = df_2.iloc[:,0]
    
    import numpy as np
    mae = pd_serie1 - pd_serie2
    sd = mae.apply(lambda x: np.abs(x)).std()
    mae = mae.apply(lambda x : np.abs(x)).mean()

    return mae, sd

## entrega el test chi2 de independencia para variable categoricas
def chi2_category(pd_df):
    # pd_df = geo15[['Cargado', 'Destino', 'Equipo', 'Grupo', 'Material',
    #                         'Operador', 'Origen', 'Ratio', 'Turno']]
    from scipy.stats import chi2_contingency
    corr_chi2 = {}
    p_value = []
    chi2_value = []
    df = []
    for fila in list(pd_df):
        corr_fila_1 = []
        corr_fila_2 = []
        corr_fila_3 = []
        for columns in list(pd_df):
            test = chi2_contingency(pd.crosstab(pd_df[fila], pd_df[columns]))
            p_value_aux = test[1]
            chi2_value_aux = test[0]
            df_aux = test[3]
            corr_fila_1.append(p_value_aux)
            corr_fila_2.append(chi2_value_aux)
            corr_fila_3.append(df_aux)
        p_value.append(corr_fila_1)
        chi2_value.append(corr_fila_2)
        df.append(corr_fila_3)
        
    corr_chi2['p-value'] = pd.DataFrame(p_value, columns = list(pd_df), index = list(pd_df))
    corr_chi2['chi2_value'] = pd.DataFrame(chi2_value, columns = list(pd_df), index = list(pd_df))
    corr_chi2['degrees_freedom'] = pd.DataFrame(df, columns = list(pd_df), index = list(pd_df))

    return corr_chi2

def report_duplicated(pd_df):
    for var in list(pd_df):
        print(var, ' ------ ',pd_df[var].duplicated().sum())

# crea un df con las variables elevadas a la potencia dada en "exponente"
def elevator(pd_, x, y=None, exponente=7):
    
    # pd_
    # x=['alimentacion_pulpa_7',
    #                 'descarga_pulpa_7',
    #                 'floculante_7']

    X = pd.DataFrame()
    covariables = ''
    for v in x:
    
        x_ = pd.DataFrame()
        name_col = []
        for e in range(1, exponente+1):
            x_ = pd.concat([x_, pd_[v]**e], axis=1)
            if e == 1:
                name_col.append(v)
            else:
                name_col.append(v+'_'+str(e))
        x_.columns = name_col
        
        X = pd.concat([X, x_], axis=1)
        name_col = pd.Series(name_col)
        name_col = name_col+' + '
    
        for e in name_col:
            covariables = covariables + e
    covariables = covariables[:-3]
    
    if y!=None:
        X = pd.concat([X, pd_[y]], axis=1)
    
    return X