import sys
sys.path.append("/opt")
import boto3
from pandas import DataFrame, read_pickle, Series
from numpy import abs, nan, dot
from time import time
from datetime import datetime
from urllib.parse import unquote_plus
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import kstest_normal

s3 = boto3.client('s3')


def mae(pd_serie1, pd_serie2):

    pd_serie1 = pd_serie1.copy()
    pd_serie2 = pd_serie2.copy()
    
    df_1 = DataFrame(pd_serie1)
    df_2 = DataFrame(pd_serie2)
    
    df_1.reset_index(drop = True, inplace=True)
    df_2.reset_index(drop = True, inplace=True)

    pd_serie1 = df_1.iloc[:,0]
    pd_serie2 = df_2.iloc[:,0]
    
    mae = pd_serie1 - pd_serie2
    sd = mae.apply(lambda x: abs(x)).std()
    mae = mae.apply(lambda x : abs(x)).mean()

    return mae, sd

def mse(pd_serie1, pd_serie2):
    pd_serie1 = pd_serie1.copy()
    pd_serie2 = pd_serie2.copy()
    df_1 = DataFrame(pd_serie1)
    df_2 = DataFrame(pd_serie2)
    
    df_1.reset_index(drop = True, inplace=True)
    df_2.reset_index(drop = True, inplace=True)

    pd_serie1 = df_1.iloc[:,0]
    pd_serie2 = df_2.iloc[:,0]
    
    mse = pd_serie1 - pd_serie2
    
    mse = mse.apply(lambda x : x**2).mean()
    return mse

def check_normal(pd_serie, bins = None):
    
    name = pd_serie.name
    
    if isinstance(name, (str)) != True:
        name = ''
    
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

## funcion que predice un conjunto test temporal dado el
## modelo var mod_var de statsmodels
## k : posicion de la variable respuesta
def prediccion(test, mod_var, k = 0):
    # mod_var = var_model
    # test = test_
    # k = 1
    
    p = mod_var.k_ar
    coefs = [mod_var.intercept[k]]
    for i in range(mod_var.k_ar):
        coefs = coefs + mod_var.coefs[i][k].tolist()
    
    predicciones_potencia = [nan]*p
    for i in range(0,len(test)-p):
        # i = 0
        rezago_aux = test.iloc[i:i+p].values.tolist()
        rezagos = []
        # cols info anteiores
        for rezago_t in reversed(rezago_aux):
            rezagos = rezagos + rezago_t
        rezagos = [1] + rezagos
        predicciones_potencia.append(dot(coefs, rezagos))

    return predicciones_potencia

def lambda_handler(event, context):

    """
    Función que entrena un modelo VAR con la libreria statmodels
    """


    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = unquote_plus(event['Records'][0]['s3']['object']['key'])
    s3.download_file(bucket_name,key,'/tmp/' + key)
    sag_estacionario = read_pickle('/tmp/' + key)
    #sag_estacionario = read_pickle('sag_estacionario.pkl')

    
    train = sag_estacionario[(sag_estacionario.index >= datetime(2020,1,1)) &\
                        (sag_estacionario.index < datetime(2020,4,1))]
    train.reset_index(inplace = True)
    train.sort_values('fecha', inplace = True)

    test = sag_estacionario[(sag_estacionario.index >= datetime(2020, 4, 1)) & \
                        (sag_estacionario.index < datetime(2020, 5, 1))]
    test.reset_index(inplace = True)
    test.sort_values('fecha', inplace = True)

    # =============================================================================
    # Modelo
    # =============================================================================
    ## potencia
    # for i in range(1,6):

    info_modelo = {}
    p = 8
    y = 'potencia'
    # covariables = ['tph', 'velocidad', 'agua']

    # prediccion
    covariables = ['tph', 'velocidad', 'agua', 'porcSolido', 'gruesos',
                   'finos', 'medios', 'e_especifica']

    cov_name_formula = []
    for i in range(p):
        cov_name_t = Series([y] + covariables) + '_t-' + str(i+1)
        cov_name_t = cov_name_t.values.tolist()
        cov_name_formula = cov_name_formula + cov_name_t

    ads_train = train[[y] + covariables]

    info_modelo['variables_usadas'] = list(ads_train)
    idx_date_train = train['fecha']
    info_modelo['fecha_train_inicio'] = idx_date_train.min()
    info_modelo['fecha_train_final'] = idx_date_train.max()

    test_ = test[list(ads_train)]
    idx_date_test = test['fecha']
    info_modelo['fecha_test_inicio'] = idx_date_test.min()
    info_modelo['fecha_test_final'] = idx_date_test.max()

    
    model = VAR(ads_train)

    info_modelo['demora'] = []
    start_time = time()
    var_model = model.fit(p)
    time_demora = time() - start_time
    info_modelo['demora'] = time_demora
    print('time train VAR'+str(p)+':',time_demora, ' seg') # tiempo de demora

    var_model.summary()
    print(var_model.aic)

    ## evaluacion modelo
    ## check supuestos
    check_normal(Series(var_model.resid[y])) ## normal
    info_modelo['info_residuos'] = check_normal(Series(var_model.resid[y]))
    print('ljung: ',acorr_ljungbox(var_model.resid[y])[1][9])
    info_modelo['ljung_test'] = acorr_ljungbox(var_model.resid[y])[1][9]


    ## evaluacion data test
    p = var_model.k_ar
    pred_test = prediccion(test_, var_model) ### hacer esta funcion y que ingrese los ultimos 4 (ingresan 5)
    pred_train = prediccion(ads_train, var_model)

    mae_train = mae(Series(pred_train),ads_train[y])
    mse_train = mse(Series(pred_train),ads_train[y])

    mae_test = mae(Series(pred_test),test_[y])
    mse_test = mse(Series(pred_test),test_[y])

    print('potencia train VAR'+str(p))
    print('mae :', mae_train[0])
    print('mae (sd):', mae_train[1])
    print('mse :', mse_train)

    print('potencia test VAR'+str(p))
    print('mae :', mae_test[0])
    print('mae (sd):', mae_test[1])
    print('mse :', mse_test)

    info_modelo['mae_train'] = mae_train[0]
    info_modelo['mae_train (sd)'] = mae_train[1]
    info_modelo['mse_train'] = mse_train
    info_modelo['mae_test'] = mae_test[0]
    info_modelo['mae_test (sd)'] = mae_test[1]
    info_modelo['mse_test'] = mse_test

    # export informacion
    #with open('modelos/m_var'+str(p)+'_'+y+'_info.txt1', 'w') as file:
    #    file.write("info_model = { \n")
    #    for k in sorted (info_modelo.keys()):
    #        file.write("'%s':'%s', \n" % (k, info_modelo[k]))
    #    file.write("}")

    # export model
    #var_model.save('modelos/m_var'+str(p)+'_'+y+'1.pkl')

    # export pesos

    #coefs = [var_model.intercept[0]]
    #for i in range(var_model.k_ar):
    #    print(i)
    #    coefs = coefs + var_model.coefs[i][0].tolist()

    #try:
    #    # si falla es porque se agrego intercepto
    #    coefs = pd.DataFrame({'coefs':coefs, 'variable': [cov_name_formula]})
    #except :
    #    coefs = pd.DataFrame({'intercept+coef':coefs,
    #                          'variable': ['intercept'] + cov_name_formula})
    #    
    #coefs.to_excel('modelos/m_var'+str(p)+'_'+y+'_coefs1.xlsx')