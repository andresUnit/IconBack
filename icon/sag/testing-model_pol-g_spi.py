import pandas as pd
import numpy as np
import datetime as dt
pd.set_option('max.columns', 100)

test = pd.read_pickle('../../../../datos/sag_estacionario.pkl')
test = test.loc[dt.datetime(2020, 2, 14):]
test = test.dropna()


results_list = {}

results_list['mean_potencia'] = []
results_list['mean_presion'] = []
results_list['mean_pebbles'] = []

results_list['std_potencia'] = []
results_list['std_presion'] = []
results_list['std_pebbles'] = []

results_list['mae_potencia'] = []
results_list['mae_presion'] = []
results_list['mae_pebbles'] = []

results_list['mse_potencia'] = []
results_list['mse_presion'] = []
results_list['mse_pebbles'] = []

results_list['rmse_potencia'] = []
results_list['rmse_presion'] = []
results_list['rmse_pebbles'] = []

results = test[['potencia', 'presion', 'pebbles']]

results['sim_potencia'] = np.nan
results['sim_presion'] = np.nan
results['sim_pebbles'] = np.nan
casos = []

for i in range(len(test)):
    print(i/len(test)*100)
    input_ = test.iloc[i][['tph', 'finos', 'medios', 'gruesos',
                           'velocidad', 'agua']]
    input_ = pd.DataFrame(input_).transpose()

    input_spi = test.iloc[i][['spi_round_idx']].values[0]
    input_spi = int(input_spi)

    # abrimos modelo por spi correspondiente (spi_input)
    for spi in np.arange(38,53).tolist():
        if input_spi==spi:
            m_sim = np.load('modelos/mod'+str(input_spi)+'_objeto.npy',
                            allow_pickle=True)
            aux = input_spi
        else:
            pass

    try:
        print('model choise: ', aux)
    
        test = pd.DataFrame(test)
    
        X = pd.DataFrame()
        covariables = ''
        for v in list(input_):
    
            x = pd.concat([input_[v], input_[v]**2, input_[v]**3], axis=1)
            x.columns = [v, v+'_2', v+'_3']
            covariables = covariables+' + '+v+' + '+v+'_2 + '+v+'_3'
            
            X = pd.concat([X, x], axis=1)
    
        results['sim_potencia'].iloc[i] = m_sim[0][1].predict(X)[0]
        results['sim_presion'].iloc[i] = m_sim[1][1].predict(X)[0]
        results['sim_pebbles'].iloc[i] = m_sim[2][1].predict(X)[0]
        
        del m_sim
        del aux
        
    except:
        print('model not found')


results['error_potencia'] = results['potencia'] - results['sim_potencia']
results['error_presion'] = results['presion'] - results['sim_presion']
results['error_pebbles'] = results['pebbles'] - results['sim_pebbles']

results['aerror_potencia'] = np.abs(results['error_potencia'])
results['aerror_presion'] = np.abs(results['error_presion'])
results['aerror_pebbles'] = np.abs(results['error_pebbles'])

results['serror_potencia'] = results['error_potencia']**2
results['serror_presion'] = results['error_presion']**2
results['serror_pebbles'] = results['error_pebbles']**2

pd.concat([test[['tph', 'finos', 'medios', 'gruesos', 'velocidad',
                 'porcSolido', 'agua', 'spi_round_idx', 'spi_round',
                 'spi']], results], axis=1).\
    to_excel('../../datos/results_mod_SPI.xlsx')

results_list['mae_potencia'].append(results['aerror_potencia'].mean())
results_list['mae_presion'].append(results['aerror_presion'].mean())
results_list['mae_pebbles'].append(results['aerror_pebbles'].mean())

results_list['mse_potencia'].append(results['serror_potencia'].mean())
results_list['mse_presion'].append(results['serror_presion'].mean())
results_list['mse_pebbles'].append(results['serror_pebbles'].mean())

results_list['rmse_potencia'].\
    append(np.sqrt(results['serror_potencia'].mean()))
results_list['rmse_presion'].\
    append(np.sqrt(results['serror_presion'].mean()))
results_list['rmse_pebbles'].\
    append(np.sqrt(results['serror_pebbles'].mean()))

results_list['std_potencia'].append(test['potencia'].std())
results_list['std_presion'].append(test['presion'].std())
results_list['std_pebbles'].append(test['pebbles'].std())

results_list['mean_potencia'].append(test['potencia'].mean())
results_list['mean_presion'].append(test['presion'].mean())
results_list['mean_pebbles'].append(test['pebbles'].mean())


results_list = pd.DataFrame(results_list)

results_list['%mae_potencia'] = results_list['mae_potencia']/ \
    results_list['mean_potencia']*100
results_list['%mae_presion'] = results_list['mae_presion']/ \
    results_list['mean_presion']*100
results_list['%mae_pebbles'] = results_list['mae_pebbles']/ \
    results_list['mean_pebbles']*100

results_list.index = [dt.datetime(2020, 2, 14)]

results_list.to_excel('modelos/polinomial_spi_info_results.xlsx')
