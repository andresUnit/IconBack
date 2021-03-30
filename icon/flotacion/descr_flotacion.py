import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from funciones.stats_code import (descriptivo)
from funciones.f_series import grupos_densidad
pd.set_option('max.columns', 100)
sns.set_style('whitegrid')

df = pd.read_excel("../../../datos/dataflotacion.xlsx")

df1 =  df[["Fecha", "Recuperacion Global","%Cu Conc final","%Cu Cola final",
             "%Cu Alimen. Rougher","%Fe Alimen. Rougher",
             "%Sol Alimen. Rougher","PH Rougher","%Cu colas Rougher",
             "%Fe colas Rougher","%Sol colas Rougher","%Cu Conc. Rougher",
             "%Cu Concentrado limpieza Rougher","%FeConcentrado limpieza Rougher",
             "%Sol Concentrado limpieza Rougher", "DI-101", "Espumante STD",
             "Xantato", "NaHS", 'Tph tratamiento']]
df1.set_index('Fecha', inplace=True)


# del outlier: con p1 - p99
df1 = df1[(df1["%Cu Conc final"]>=16) & (df1["%Cu Conc final"]<=40)]
df1 = df1[(df1["Recuperacion Global"]>=70) & (df1["Recuperacion Global"]<=95)]
df1 = df1[(df1["%Cu Alimen. Rougher"]>=0.54) & (df1["%Cu Alimen. Rougher"]<=1.5)]
df1 = df1[(df1["%Cu Cola final"]>=0.09) & (df1["%Cu Cola final"]<=0.22)]
df1 = df1[(df1["DI-101"]>=14) & (df1["DI-101"]<=40)]
df1 = df1[(df1["Xantato"]>0) & (df1["Xantato"]<=4.5)]
df1 = df1[(df1["Espumante STD"]>=5) & (df1["Espumante STD"]<=28)]
df1 = df1[(df1["PH Rougher"]>=10) & (df1["PH Rougher"]<=11)]
df1 = df1[(df1['%Cu Conc. Rougher']>=0.09) & (df1['%Cu Conc. Rougher']<=13.2)]
df1 = df1[(df1['%Sol colas Rougher']>=23.21) & (df1['%Sol colas Rougher']<=33.27)]

# nuevas variables
df1['Razon Cu/Fe'] = df1['%Cu Alimen. Rougher']/df1['%Fe Alimen. Rougher']
df1['variacion rec'] = df1['Recuperacion Global'].shift(1) - df1['Recuperacion Global']
df1['variacion conc'] = df1['%Cu Conc final'].shift(1) - df1['%Cu Conc final']
df1['variacion colafinal'] = df1['%Cu Cola final'].shift(1) - df1['%Cu Cola final']


# =============================================================================
# Descriptivo
# =============================================================================

for v in list(df1):
    plt.figure(figsize=(10,3))
    plt.plot(df1[v])
    plt.title(v)
    plt.show()

for v in list(df1):
    plt.figure(figsize=(8,8))
    sns.distplot(df1[v], bins=50)
    plt.show()


# del outlier: con p99
# df1 = df1[df1['%FeConcentrado limpieza Rougher']<28.54]
# df1 = df1[df1['%Fe Alimen. Rougher']<2.5]
# df1 = df1[df1['Razon Cu/Fe']<0.70]
# df1 = df1[df1['Espumante STD']<26.93]
# df1 = df1[df1['%Sol Concentrado limpieza Rougher']<38.23]
# df1 = df1[df1['%Cu Cola final']<0.4]
# df1 = df1[df1['Xantato']<10]
# df1 = df1[df1['%Cu Alimen. Rougher']>0.35]
# df1 = df1[df1['DI-101']<45]


print('Datos nulos desde 27 de Agosto del 2019 a las 15:10:00 para %Cu y %Solido \
      en Concentrado Limpieza Rougher')

# reporte
descriptivo(df1).to_excel('reportes/estadisticas_flotacion.xlsx')

# save
# df1.to_pickle('../../../datos/dataflotacion2.pkl')

# =============================================================================
# dispersiones
# =============================================================================
#load
# df1 = pd.read_pickle('../../../datos/dataflotacion2.pkl')

# grupos 2d
from funciones.stats_code import grupos_densidad_2d
g_ = grupos_densidad_2d(df1['%Cu Conc final'], df1['Recuperacion Global'],
                           kmean_parameters=[3, 'k-means++', 100, 500])
df1['g_CuAlimRou/RecupGlob'] = g_

# =============================================================================
# Rangos
# =============================================================================

# %Cu Conc final
g_, _1 = grupos_densidad(df1['%Cu Conc final'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)
df1['g_CuConcfinal'] = g_
df1['g_CuConcfinal'][df1['g_CuConcfinal']==0] = 'min'
df1['g_CuConcfinal'][df1['g_CuConcfinal']==1] = 'medio'
df1['g_CuConcfinal'][df1['g_CuConcfinal']==2] = 'max'

# Recuperacion Global
g_, _2 = grupos_densidad(df1['Recuperacion Global'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_RecupGlob'] = g_
df1['g_RecupGlob'][df1['g_RecupGlob']==0] = 'min'
df1['g_RecupGlob'][df1['g_RecupGlob']==1] = 'medio'
df1['g_RecupGlob'][df1['g_RecupGlob']==2] = 'max'


# %Cu Alimen. Rougher
g_, _3 = grupos_densidad(df1['%Cu Alimen. Rougher'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)
df1['g_CuAlimRou'] = g_
df1['g_CuAlimRou'][df1['g_CuAlimRou']==0] = 'min'
df1['g_CuAlimRou'][df1['g_CuAlimRou']==1] = 'medio'
df1['g_CuAlimRou'][df1['g_CuAlimRou']==2] = 'max'

# %Cu Cola Final
g_, _4 = grupos_densidad(df1['%Cu Cola final'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_cu_colafinal'] = g_
df1['g_cu_colafinal'][df1['g_cu_colafinal']==0] = 'min'
df1['g_cu_colafinal'][df1['g_cu_colafinal']==1] = 'medio'
df1['g_cu_colafinal'][df1['g_cu_colafinal']==2] = 'max'

g_, _5 = grupos_densidad(df1['DI-101'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_di101'] = g_
df1['g_di101'][df1['g_di101']==0] = 'min'
df1['g_di101'][df1['g_di101']==1] = 'medio'
df1['g_di101'][df1['g_di101']==2] = 'max'

g_, _6 = grupos_densidad(df1['Xantato'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_xantato'] = g_
df1['g_xantato'][df1['g_xantato']==0] = 'min' 
df1['g_xantato'][df1['g_xantato']==1] = 'medio'
df1['g_xantato'][df1['g_xantato']==2] = 'max'

g_, _7 = grupos_densidad(df1['Espumante STD'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_espumante'] = g_
df1['g_espumante'][df1['g_espumante']==0] = 'min'
df1['g_espumante'][df1['g_espumante']==1] = 'medio'
df1['g_espumante'][df1['g_espumante']==2] = 'max'

g_, _8 = grupos_densidad(df1['PH Rougher'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)

df1['g_ph'] = g_
df1['g_ph'][df1['g_ph']==0] = 'min'
df1['g_ph'][df1['g_ph']==1] = 'medio'
df1['g_ph'][df1['g_ph']==2] = 'max'


# %Cu Conc Rougher
g_, _9 = grupos_densidad(df1['%Cu Conc. Rougher'], kmean_parameters=[3, 'k-means++', 100, 500],
                        seed=2019)
df1['g_CuConcRou'] = g_
df1['g_CuConcRou'][df1['g_CuConcRou']==0] = 'min'
df1['g_CuConcRou'][df1['g_CuConcRou']==1] = 'medio'
df1['g_CuConcRou'][df1['g_CuConcRou']==2] = 'max'


# save
# df1.to_pickle('../../../datos/dataflotacion3.pkl')

# load
# df1 = pd.read_pickle('../../../datos/dataflotacion3.pkl')

excluyentes = ['g_CuConcRou', 'g_RecupGlob', 'g_CuAlimRou', 'g_cu_colafinal',
               'g_di101', 'g_xantato', 'g_espumante', 'g_ph', 'g_CuConcfinal']

# =============================================================================
# Analisis de dispersion y relacion
# =============================================================================

# corregimos outlier Razon Cu/Fe: llenamos con la media
g_, _ = grupos_densidad(df1['Razon Cu/Fe'], kmean_parameters=[7, 'k-means++', 100, 500],
                        seed=2019)
df1['g_outlier'] = g_

df1['Razon Cu/Fe'][df1['g_outlier'].isin([6])] = np.nan
from funciones.f_series import mean_between
df1['Razon Cu/Fe'] = mean_between(df1['Razon Cu/Fe'])


variables = ['Recuperacion Global', '%Cu Conc final', '%Cu Cola final',
          '%Cu Alimen. Rougher','%Fe Alimen. Rougher', 'DI-101', 'Espumante STD',
          'Xantato', 'NaHS', 'Razon Cu/Fe', 'Tph tratamiento', 'PH Rougher',
          ]

# correlacion
var_nivel = 'g_CuAlimRou'
for nivel in df1[var_nivel].unique():
    pd_ = df1[df1[var_nivel]==nivel]
    pd_ = pd_[variables]
    corr = pd_.corr()
    corr2 = corr.loc[['Recuperacion Global', '%Cu Conc final']]
    # corr2.to_excel('reportes/corr_'+var_nivel+'_'+nivel+'.xlsx')
    
    plt.figure(figsize = (9, 9))
    sns.heatmap(corr2, annot = True, center = 0, square=True)
    plt.title(var_nivel+' nivel: '+nivel, fontsize=25)

y = 'Recuperacion Global'
# caso medio-max
for var in ['%Cu Alimen. Rougher', '%Cu Cola final', 'Razon Cu/Fe',
            'NaHS', 'PH Rougher', 'DI-101']:
    
    pd_ = df1[df1[var_nivel].isin(['max'])]
    plt.figure(figsize=(15,15))
    ax = sns.scatterplot(var, y, hue = var_nivel, data = pd_)
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    plt.xlabel(var, fontsize=25)
    plt.ylabel(y, fontsize=25)
    plt.show()

y = '%Cu Conc final'
# caso medio-max
for var in ['%Fe Alimen. Rougher', '%Cu Alimen. Rougher', 'Razon Cu/Fe',
            'NaHS']:
    
    pd_ = df1[df1[var_nivel].isin(['min'])]
    plt.figure(figsize=(15,15))
    ax = sns.scatterplot(var, y, hue = var_nivel, data = pd_)
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    plt.xlabel(var, fontsize=25)
    plt.ylabel(y, fontsize=25)
    plt.show()
