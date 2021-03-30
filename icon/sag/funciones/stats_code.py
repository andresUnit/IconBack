import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## analisis de tendencia central, posición relativa y disperción
## var_omitir: nombre de variable del pd que no se quiere retornar un descriptivo
def descriptivo(df_pd, var_omitir = None, cuantiles = np.arange(0,1.1,0.1)):
    
    # df_pd = espesadores.copy()
    
    df_pd = df_pd.copy()
    df_pd = pd.DataFrame(df_pd)

    if var_omitir != None:
        for var in var_omitir:
            del df_pd[var]
            
    # covnertir a float
    try:
        df_pd = df_pd.astype(float)
    except:
        raise ValueError('Variables no float, use var_omitir para omitir var no float')

    desc = pd.DataFrame()
    desc['count'] = df_pd.count()
    desc['mean'] = df_pd.mean()
    desc['median'] = df_pd.median()
    desc['std'] = df_pd.std()
    desc['var'] = df_pd.var()
    desc['min'] = df_pd.min()
    desc['max'] = df_pd.max()
    desc['rango'] = desc['max'] - desc['min']

    cuantiles = df_pd.describe(cuantiles).iloc[4:-1].transpose()
    desc.reset_index(inplace = True)
    cuantiles.reset_index(inplace = True)
    desc = desc.merge(cuantiles, how = 'outer', on = 'index')
    
    desc.set_index('index', inplace = True)
    
    return desc

def desc_grupos(pd_, var, var_grupos):
    desc = pd.DataFrame()
    cols = []
    for i in pd_[var_grupos].unique():
        desc = pd.concat([desc,
                          descriptivo(pd_[pd_[var_grupos]==i]).loc[var,:]],
                          axis=1)
        cols.append(i)
    desc.columns = cols
    return desc


def check_2samples(pd_serie1, pd_serie2, title = None):
    
    # pd_serie1 = sag['potencia_ls_l1']
    # pd_serie2 = sag['potencia_ls_l2']

    ## check normalidad
    pd_serie1.hist(bins = 50)
    plt.title(title)
    plt.show()
    pd_serie2.hist(bins = 50)
    plt.title(title)
    plt.show()

    
    #qqplot
    from statsmodels.graphics.gofplots import qqplot
    qqplot(pd_serie1, line = 'r')
    plt.title('qqplot pd_serie1')
    plt.show()

    qqplot(pd_serie2, line = 'r')
    plt.title('qqplot pd_serie2')
    plt.show()
    
    # tests results
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import kstest_normal
    
    print('results normal tests')
    print('pd_serie1')
    print('shapiro p-value', shapiro(pd_serie1)[1])
    print('KS p-value', kstest_normal(pd_serie1, dist = 'norm')[1])
    print()

    print('pd_serie2')
    print('shapiro p-value', shapiro(pd_serie2)[1])
    print('KS p-value', kstest_normal(pd_serie2, dist = 'norm')[1])
    print()
    ## chequeo independencia
    
    ## correlations
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    print('')
    print('Pearson Correlation:', pearsonr(pd_serie1, pd_serie2)[0])
    print('Spearman Correlation:', spearmanr(pd_serie1, pd_serie2)[0])
    plt.plot(pd_serie1, pd_serie2, '.')
    plt.title('distribucion ')
    
def check_normal(pd_serie, bins = None):
    
    # pd_serie = pd.Series(mod1.resid_pearson)
    # pd_serie = sag_l1['bwi_ls_l1']
    name = pd_serie.name
    
    if isinstance(name, (str)) != True:
        name = ''

    ## check normalidad
    sns.distplot(pd_serie, bins = bins)
    if name != '':
        plt.title(name)
    plt.show()
    
    #qqplot
    from statsmodels.graphics.gofplots import qqplot
    # import matplotlib.pyplot as plt
    qqplot(pd_serie, line = 'r')
    plt.title('qqplot '+name)
    plt.show()
    
    # tests results
    from scipy.stats import shapiro
    from statsmodels.stats.diagnostic import kstest_normal
    
    shapiro_scipystats = shapiro(pd_serie)
    ktest_statsmodelsstatsdiagnostic = kstest_normal(pd_serie, dist = 'norm')
    
    print('results normal tests', name)
    print('shapiro p-value', shapiro_scipystats[1])
    print('KS p-value', ktest_statsmodelsstatsdiagnostic[1])
    print()
    
    print('Test used scipy.stats.shapiro and statsmodels.stats.diagnostic.ktest_normal')
    print()
    
    return {'shapiro': shapiro_scipystats,
            'ktest': ktest_statsmodelsstatsdiagnostic}

    
def reg_plot(pd_serie_x, pd_serie_y, loc = 4, plt_figure = None):
    
    # pd_serie_x = sag['usoPotn_ls_l1']
    # pd_serie_y = sag['porcSolido_ls_l1']
    
    name_x = pd_serie_x.name
    name_y = pd_serie_y.name

    from sklearn.linear_model import LinearRegression
    x = pd.DataFrame(pd_serie_x)
    y = pd.DataFrame(pd_serie_y)
    lr = LinearRegression()
    reg = lr.fit(x, y)
    y_pred = reg.predict(x)
    
    
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    print('Intercept: ', reg.intercept_)
    print('coef: ',reg.coef_)
    print('mse: ', mean_squared_error(y, y_pred))
    print('R2: ', r2_score(y, y_pred))

    from seaborn import regplot
    if plt_figure != None:
        plt.figure(figsize = (10,10))
    ax = regplot(x, y, scatter_kws={'color': 'black', 's':4},
                line_kws={'color': 'red'}, ci = 95)
    
    at = textonly(plt.gca(), 'Intercept: ' + str(reg.intercept_) + '\
                            \nCoefs: ' + str(reg.coef_) + ' \
                            \nmse: ' + str(mean_squared_error(y, y_pred)) + '\
                            \nR2: ' + str(r2_score(y, y_pred)),
                            loc = loc, fontsize = 15)
    #plt.gcf().show()
    
    
## xticks : lista que indica los xticks
## dbscan_parameters: debe ser una lista [radio, n_vecinos]
## kmeans_parameters: debe ser una lsita [n_clusters, init, n_init, max_iter]
    ## puede ser por ejemplo: [4, 'k-means++', 100, 500] * defectos de sklearn
## legend : tiene que ser una lista
def grupos_densidad_2d(pd_serie1, pd_serie2, dbscan_parameters = None,
                    kmean_parameters = None,
                    xticks = None, title = None, legend = None, bins = 50,
                    reporte_path = None,
                    boxplot_extremos = True,
                    only_outlier_dbscan = False,
                    only_majority_groups = False,
                    plt_figure = None):

    # pd_serie1 = sag_estacionario['potencia']
    # pd_serie2 = sag_estacionario[y]

    # dbscan_parameters = [500, 3]
    # xticks = ['A', 'B', 'C', 'D']
    
    if dbscan_parameters == None and kmean_parameters == None:
        raise TypeError("indique parametros para 'dbscan_parameters' o 'kmean_parameters'")
    
    if dbscan_parameters != None and kmean_parameters != None:
        raise TypeError('No puede usar kmeans y dbscan al mismo tiempo')
    
    name_serie1 = pd_serie1.name
    name_serie2 = pd_serie2.name

    if (pd_serie1.isnull().sum() > 0) or (pd_serie2.isnull().sum() > 0) :
        raise ValueError('Series de pandas contiene Nans')
    try:
        pd_serie1 = pd_serie1.astype(float)
        pd_serie2 = pd_serie2.astype(float)
    except:
        raise TypeError('Imposible convertir a float valores de las Series de pandas')
    
    pd_serie = pd.concat([pd_serie1, pd_serie2], axis = 1)
    
    if dbscan_parameters != None:
        from sklearn.cluster import DBSCAN
        X = pd_serie.values
        clusters = DBSCAN(dbscan_parameters[0], dbscan_parameters[1])
        groups = clusters.fit(X)
        pd_serie['groups'] = groups.labels_
    
    if kmean_parameters != None:
        from sklearn.cluster import KMeans
        X = pd_serie.values
        clusters = KMeans(n_clusters = kmean_parameters[0],
                          init = kmean_parameters[1],
                          n_init = kmean_parameters[2],
                          max_iter = kmean_parameters[3])
        groups = clusters.fit(X)
        pd_serie['groups'] = groups.labels_
        
    if only_outlier_dbscan == True and only_majority_groups == True:
        raise TypeError('only_outlier_dbscan y only_majority_groups no deben ser True al mismo tiempo')
    
    if only_outlier_dbscan == True and kmean_parameters == None:
        pd_serie['groups'][pd_serie['groups'] != -1] = 0
        
    if only_majority_groups == True and kmean_parameters == None:
        pd_serie['groups'][pd_serie['groups'] != 0] = -1
        
    if plt_figure != None: 
        plt.figure(figsize=(10,10))
        
    sns.scatterplot(x = name_serie1, y = name_serie2,
            data = pd_serie, hue = 'groups',
            palette = 'colorblind')
    
    return pd_serie['groups']



## funcion que permite agregar legend text
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

## ax tiene que ser un obj plt.gca()
def textonly(ax, txt, fontsize = 14, loc = 2, *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize), 
                      frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at

## LA FUNCION SIGUIENTE ESTA ANCLADA A LA ANTERIOR
## genera modelos de reg polinomial (lineal mayor a 2) de una matriz de
## variables. Esta matriz debe contener a la variable respuesta
## retorna los modelos en mod_pol (tipo sklearn) e informacion de ellos en info_pol

# def reg_pol(pd_df, y_name, grade = 2):

#     pd_df = pd_.copy()
#     y_name = 'bwi_ls_l1'
#     grade = 4

#     info_pol = pd.DataFrame()
#     mod_pol= {}

#     # grade = 4
#     # info_pol = pd.DataFrame()
#     # mod_pol = {}
#     for v in list(pd_df):

#         v = list(pd_)[3]

#         x = pd_df[v][~pd_df[v].isnull()]
#         y_ = pd.DataFrame(pd_df[y_name][~pd_df[v].isnull()])

#         x_ = x.copy()

#         for g in range(2, grade + 1):
#             x = pd.concat([x, x_**g], axis = 1)

#         # print(x)

#         x.reset_index(drop = True, inplace = True)
#         y_.reset_index(drop = True, inplace = True)
#         from sklearn.linear_model import LinearRegression
#         mod = LinearRegression()
#         lineal = mod.fit(x, y_)
#         y_pred = lineal.predict(x)
#         mod_pol[v] = lineal

#         from sklearn.metrics import r2_score
#         r2 = {'R2':[r2_score(y_, y_pred)],
#               'intercept':[lineal.intercept_[0]]}
#         for g in range(1, grade + 1):
#             r2['coef_x'+str(g)] = [lineal.coef_[0][g - 1]]

#         r2 = pd.DataFrame(r2)

#         # r2 = pd.DataFrame({'R2':[r2_score(y_, y_pred)],
#         #                    'intercept':[lineal.intercept_[0]],
#         #                    'coef_x':[lineal.coef_[0][0]],
#         #                    'coef_x2':[lineal.coef_[0][1]]})
#         r2.index = [v]
#         info_pol = pd.concat([info_pol, r2])

#     return info_pol, mod_pol



def reg_pol(pd_df, y_name, grade = 2):
    
    # pd_df = pd_.copy()
    # y_name = 'bwi_ls_l1'
    # grade = 1
    y_name = str(y_name)
    info_pol = pd.DataFrame()
    mod_pol= {}

    for v in list(pd_df):

        # v = list(pd_)[4]

        if v == y_name:
            continue

        x = pd_df[v][~pd_df[v].isnull()]
        x = pd.DataFrame(x)
        y_ = pd.DataFrame(pd_df[y_name][~pd_df[v].isnull()])
        
        x_ = x.copy()
        
        if grade > 1:
            for g in range(2, grade + 1):
                x[str(v)+'_'+str(g)] = x_**g
        else:
            pass

        x[y_name] = y_
        
        del x_; del y_

        # make formula
        formula = y_name + ' ~ '
        for covar in x.columns[:-1]:
            formula = str(formula) + str(covar) + ' + '
        formula = formula[:-3]

        print(formula)
        from statsmodels.formula.api import ols
        mod = ols(formula, data = x)
        mod = mod.fit()
        mod_pol[v] = mod

        coef = []
        for coef_number in range(1,grade+1):
            coef.append('coef_' + str(coef_number))
        coef = ['intercept'] + coef
        
        r2 = mod.params.to_dict()
        r2['R2'] = [mod.rsquared]
        r2 = pd.DataFrame(r2)
        r2.index = [v]
        r2 = r2[[list(r2)[-1]] + list(r2)[:-1]]
        r2.columns = ['R2'] + coef
   
        info_pol = pd.concat([info_pol, r2])
        
    return info_pol, mod_pol

def forward_selected(data, response, choise_criteria = 'R2', alpha = 0.05,
                     records=False):
    
    # data = X
    # response = y
    import statsmodels.formula.api as smf
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    records_ = {}
    records_out = {'R2':[], 'sign_level':[]}
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            if choise_criteria == 'R2':
                m = smf.ols(formula, data).fit()
                score = m.rsquared
                pval = m.pvalues
                mask = pval < alpha
                porcSig = mask.sum() / len(pval)

                if porcSig == 1:
                    # print(round(score,6), ' ***')
                    records_out['sign_level'].append(3)
                if porcSig >= 0.6 and porcSig < 1:
                    # print(round(score,6), ' **')
                    records_out['sign_level'].append(2)
                if porcSig >= 0.3 and porcSig < 0.6:
                    # print(round(score,6), ' *')
                    records_out['sign_level'].append(1)
                if porcSig < 0.3:
                    # print(round(score,6))
                    records_out['sign_level'].append('')
                if records==True:
                    records_[round(score,6)] = m
                records_out['R2'].append(round(score,6))

            if choise_criteria == 'R2-adj':
                m = smf.ols(formula, data).fit()
                score = m.rsquared_adj
                pval = m.pvalues
                mask = pval < alpha
                porcSig = mask.sum() / len(pval)
                # print('PorcSig',porcSig)
                if porcSig == 1:
                    # print(round(score,6), ' ***')
                    records_out['sign_level'].append(3)
                if porcSig >= 0.6 and porcSig < 1:
                    # print(round(score,6), ' **')
                    records_out['sign_level'].append(2)
                if porcSig >= 0.3 and porcSig < 0.6:
                    # print(round(score,6), ' *')
                    records_out['sign_level'].append(1)
                if porcSig < 0.3:
                    # print(round(score,6))
                    records_out['sign_level'].append('')
                if records==True:
                    records_[round(score,6)] = m
                records_out['R2'].append(round(score,6))

            if choise_criteria == 'R2-alpha':
                m = smf.ols(formula, data).fit()
                pval = m.pvalues
                mask = pval < alpha
                if mask.sum()!=len(pval):
                    score = 0.00001
                else:
                    score = m.rsquared
                    print(score)

            scores_with_candidates.append((score, candidate))
            
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    
    return model, records_, pd.DataFrame(records_out)

# data : data base en formato pandas
# y : texto del nombre de la variable respuesta del dataframe
# x : list de nombres de covariables de data
# grade : grado del polinomio
def pol_stepwise(data, y, x = None, grade = 2, method = 'stepwise', alpha = 0.05):

    # data = pd_
    # y = y
    # x = var
    
    if x != None:
        data = data[x + [y]]
    
    X = pd.DataFrame()
    covariables = ''
    for v in list(data):
        
        if v == y:
            continue
        
        x = pd.DataFrame()
        for i in range(1,grade+1):
            x[v+'_'+str(i)] = data[v]**i
    
        for name in list(x):
            covariables = covariables + ' + '+name
        
        X = pd.concat([X, x], axis = 1)
    
    X = pd.concat([X, data[y]], axis = 1)
    
    # polinomio stepwise
    mod_f, mod_records, mod_records_out = forward_selected(X, y, alpha = alpha)
    
    return mod_f, mod_records, pd.DataFrame(mod_records_out)

## plotea modelos de reg polinomial (obj LinearRegresor sklearn)
def plot_pol(x, y, data, mod_pol, plt_figure = True, color = None,
             color_points = None, title = True, xlabel = True, ylabel = True):
    
    # x = v
    # y = y
    # data = pd_
    # mod_pol = mod_pol2_l1[v]

    if x == y:
        raise ValueError('x e y misma variable: ' + x)
    
    grade = len(mod_pol.params) - 1
    
    x_ = data[x][~data[x].isnull()]
    y_ = pd.DataFrame(data[y][~data[x].isnull()])
    
    x_ = pd.DataFrame(x_)

    x_aux = x_.copy()
    
    if grade > 1:
        for i in range(2, grade + 1):
            x_[x + '_' + str(i)] = x_aux**i

        del x_aux

    x_.reset_index(drop = True, inplace = True)
    y_.reset_index(drop = True, inplace = True)

    y_pred = mod_pol.predict(x_)
    data_aux = pd.concat([x_.iloc[:,0], y_, pd.DataFrame(y_pred).iloc[:,0]], axis = 1)

    ## creamos el eje x (entre min x max x) para aplicar pol
    x_input = pd.DataFrame()
    x_input[x] = np.arange(data_aux[x].min(),
                        data_aux[x].max(), 0.01)
    x_input_aux = x_input.copy()

    if grade > 1:
        for i in range(2, grade + 1):
            x_input[x + '_' + str(i)] = x_input_aux**i
        del x_input_aux
    
    if plt_figure == True:
        plt.figure(figsize = (10,10))

    plt.plot(x_.iloc[:,0], y_, '.', color = color_points)
    plt.plot(x_input.iloc[:,0],
             pd.DataFrame(mod_pol.predict(x_input)).iloc[:,0],
             '--', color = color)
    if title == True:
        plt.title('pol'+str(grade)+' '+x)
    if xlabel == True:
        plt.xlabel(x)
    if ylabel == True:
        plt.ylabel(y)
