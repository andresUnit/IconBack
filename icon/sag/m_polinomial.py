## Modelo polinomial por partes del molino sag
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

def m_sim(pd_, y, x, exponente=3, choise_criteria='R2-alpha', ifnorm=False):
    
    """Función que realiza modelos de simulacion (regresion con covariables polinomiales)".
    
    Parameters
    -------------
    
    pd_: dataframe formato pandas.
    y: str que indica la variable respuesta que esta en pd_.
    x: lista con strings de las covariables que estan en pd_.
    exponente: exponente de la regresion polinomial.
    choise_criteria: {'R2-alpha', 'R2-adj', 'R2'}
        
        criterio para elegir el mejor modelo.
        * R2-alpha por defecto.
        
        R2: eleige segun el mejor R2.
        R2-adj: elige segun el mejor R2 Ajustado.
        R2-alpha: elige segun el mejor R2 de los modelos con variables significativas
        
    ifnorm: {True, False}
    
    Returns: 
    -------
        
        Tupla con el modelo con todas las covariables y el modelo elegido segun los
        criterios.
    """
    
    # y = 'potencia'

    X = pd.DataFrame()
    covariables = ''
    for v in x:
    
        x_ = pd.DataFrame()
        name_col = []
        for e in range(1, exponente+1):
            pd_[v]**e
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

    X = pd.concat([X, pd_[y]], axis=1)

    from statsmodels.formula.api import ols
    formula = y+' ~ '+covariables
    mod = ols(formula, data=X)
    mod = mod.fit()
    mod.summary()

    from funciones.stats_code import check_normal
    check_normal(mod.resid)
    
    # stepwise
    from funciones.stats_code import forward_selected
    modfs = forward_selected(X, response=y, choise_criteria=choise_criteria)
    modfs = modfs[0]

    check_normal(modfs.resid)

    return mod, modfs


def sim_gtph(sag_estacionario,
             train_date_until,
             grupos=[100, 101, 102, 103, 104, 200, 201, 300, 301, 4, 5, 6, 7, 8, 9],
             version_model=2):
    
    """Función que crea los modelos simulación en base a los grupos tph para
       potencia, presion, y pebbles.

    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacionaria y
                      columna "grupos_situaciones".
        
    train_date_until: fecha hasta donde se quiere tomar como data train a la data
                      sag_estacionario.
                      
    grupos: grupos tph definidos en g_tph que se quieren utilizar para modelar
    
            * En particular se usan los definidos en "grupos 2", pero se pueden
            utilizar los definidos en "grupos 3":
                
                grupos 2: [100, 101, 102, 103, 104,
                           200, 201,
                           300, 301,
                           4, 5, 6, 7, 8, 9]
                
                grupos 3: [100, 101, 102, 103, 104,
                           200, 201,
                           300, 301,
                           6, 6000,
                           7, 7000,
                           8, 8000,
                           9, 9000]
                
        version_model: version del modelo
        
    Returns: 
        
        No retorna nada. Exporta los modelos directamente en la ruta:
            
            modelos/mod[x].npy con x: numero de grupo
    -------
    """
    
    version_model = '_v'+str(version_model)
    
    train = sag_estacionario.loc[:train_date_until]
    train['grupos 2'] = train['grupos 2'].astype(int)

    mods = {}
    for g in grupos:
        mods['mod'+str(g)] = []
    
    # potencia
    for i in range(len(grupos)):
        
        pd_ = train[['velocidad', 'agua', 'tph', 'finos', 'gruesos', 'medios',
                     'pebbles', 'potencia', 'presion', 'porcSolido']]
        pd_ = pd_[train['grupos 2'] == grupos[i]]
        
        if len(pd_[train['grupos 2'] == grupos[i]]) < 3:
            print('no hay datos grupo:', grupos[i])
            continue
        
        # modelo polinomial [0]: mod con todos los exp
        #                   [1]: mod mejor R2 y var significativas (0.05)
        mods['mod'+str(grupos[i])].append(m_sim(pd_, 'potencia', ['tph', 'finos',
                                                                'medios', 'gruesos',
                                                                'velocidad', 'agua']))
        print('grupo ', grupos[i])
    
    # presion
    for i in range(len(grupos)):
        pd_ = train[['velocidad', 'agua', 'tph', 'finos', 'gruesos', 'medios',
                     'pebbles', 'potencia', 'presion', 'porcSolido']]
        pd_ = pd_[train['grupos 2'] == grupos[i]]
        
        if len(pd_[train['grupos 2'] == grupos[i]]) < 3:
            print('no hay datos grupo:', grupos[i])
            continue
        
        # modelo polinomial [0]: mod con todos los exp
        #                   [1]: mod mejor R2 y var significativas (0.05)
        mods['mod'+str(grupos[i])].append(m_sim(pd_, 'presion',
                                              ['tph', 'finos', 'medios', 'gruesos',
                                               'velocidad', 'agua']))
        print('grupo ', grupos[i])
    
    # pebbles
    for i in range(len(grupos)):
        pd_ = train[['velocidad', 'agua', 'tph', 'finos', 'gruesos', 'medios',
                     'pebbles', 'potencia', 'presion', 'porcSolido']]
        pd_ = pd_[train['grupos 2'] == grupos[i]]
        
        if len(pd_[train['grupos 2'] == grupos[i]]) < 3:
            print('no hay datos grupo:', grupos[i])
            continue
        
        # modelo polinomial [0]: mod con todos los exp
        #                   [1]: mod mejor R2 y var significativas (0.05)
        mods['mod'+str(grupos[i])].append(m_sim(pd_, 'pebbles', ['tph', 'finos',
                                                               'medios', 'gruesos',
                                                               'velocidad', 'agua']))
        print('grupo ', grupos[i])
    
    # save
    for i in range(len(grupos)):
        np.save('modelos/mod'+str(grupos[i])+'_objeto'+version_model+'.npy',
                mods['mod'+str(grupos[i])])
    print('modelos exportados en modelos/mod_objeto[x].npy con x: numero de grupo')

    # save pesos
    for i in range(len(grupos)):
        aux = (list(mods['mod'+str(grupos[i])][0][1].params.index),
               mods['mod'+str(grupos[i])][0][1].params.values,
               list(mods['mod'+str(grupos[i])][1][1].params.index),
               mods['mod'+str(grupos[i])][1][1].params.values,
               list(mods['mod'+str(grupos[i])][2][1].params.index),
               mods['mod'+str(grupos[i])][2][1].params.values)
        np.save('modelos/mod'+str(grupos[i])+version_model+'.npy', aux)
    print('pesos modelos exportados en modelos/mod[x].npy con x: numero de grupo')

    # # big polinom
    # pd_ = train[['velocidad', 'agua', 'tph', 'finos', 'gruesos', 'medios',
    #              'pebbles', 'potencia', 'presion', 'porcSolido']]
    # big_mod = []
    # big_mod.append(sim(pd_, 'potencia', exponente=5))
    # big_mod.append(sim(pd_, 'presion', exponente=5))
    # big_mod.append(sim(pd_, 'pebbles', exponente=5))
    
    # np.save('modelos/big_mod_objeto.npy', big_mod)

def sim_gspi(sag_estacionario, train_date_until, version_model):
    
    """Función que crea los modelos simulación segun los grupos de spi para
       potencia, presion, y pebbles.

    Parameters
    -------------
    
    sag_estacionario: df con la informacion del molino sag en situacion estacionaria y
                      columna "grupos_situaciones".
        
    train_date_until: fecha hasta donde se quiere tomar como data train a la data
                      sag_estacionario.
                      
    version_model: version del modelo
        
    Returns: 
        
        No retorna nada. Exporta los modelos directamente en la ruta:
            
            modelos/mod[x].npy con x: valor spi.
    -------
    """
    
    # version_model=1
    # train_date_until = dt.datetime(2020, 2, 14)
    
    version_model = '_v'+str(version_model)
    
    train = sag_estacionario.loc[:train_date_until]
    train['grupos 2'] = train['grupos 2'].astype(int)

    grupos_spi = sag_estacionario['spi_round'].unique().tolist()[:-1]
    grupos_spi = sorted(grupos_spi)
    grupos_spi = grupos_spi[3:-4]
    
    mods = {}
    for g in grupos_spi:
        mods['mod'+str(int(g))] = []
    
    pd_ = train[['velocidad', 'agua', 'tph', 'finos', 'gruesos', 'medios',
                  'pebbles', 'potencia', 'presion', 'porcSolido',
                  'grupos_situaciones', 'spi_round_idx', 'spi_round', 'spi']]

    ## potencia
    for i in range(len(grupos_spi)):
    
        pd_2 = pd_[pd_['spi_round_idx']==grupos_spi[i]]
        print(int(grupos_spi[i]), 'len:', len(pd_2))
        try:
            mods['mod'+str(int(grupos_spi[i]))].append(m_sim(pd_2,
                                                            'potencia',
                                                            ['tph', 'finos', 'medios'
                                                            , 'gruesos','velocidad',
                                                            'agua']))
            print('modelamiento exitoso')
        except:
            print('error modelamiento: pocos datos o ninguno')
            mods['mod'+str(int(grupos_spi[i]))] = []
        i += 1
    
    ## presion
    for i in range(len(grupos_spi)):

        pd_2 = pd_[pd_['spi_round_idx']==grupos_spi[i]]
        print(int(grupos_spi[i]), 'len:', len(pd_2))
        try:
            mods['mod'+str(int(grupos_spi[i]))].append(m_sim(pd_2,
                                                            'presion',
                                                            ['tph', 'finos', 'medios'
                                                            , 'gruesos','velocidad',
                                                            'agua']))
            print('modelamiento exitoso')
        except:
            print('error modelamiento: pocos datos o ninguno')
            mods['mod'+str(int(grupos_spi[i]))] = []
        i += 1
    
    ## pebbles
    for i in range(len(grupos_spi)):
        
        pd_2 = pd_[pd_['spi_round_idx']==grupos_spi[i]]
        print(int(grupos_spi[i]), 'len:', len(pd_2))
        try:
            mods['mod'+str(int(grupos_spi[i]))].append(m_sim(pd_2,
                                                            'pebbles',
                                                            ['tph', 'finos', 'medios'
                                                            , 'gruesos','velocidad',
                                                            'agua']))
            print('modelamiento exitoso')
        except:
            print('error modelamiento: pocos datos o ninguno')
            mods['mod'+str(int(grupos_spi[i]))] = []
        i += 1

    # save mods
    for k in mods:
        np.save('modelos/'+k+'_objeto'+version_model+'.npy',
                mods[k])

    # save mods weight
    for k in mods:
        try:
            aux = (list(mods[k][0][1].params.index),
                    mods[k][0][1].params.values,
                    list(mods[k][1][1].params.index),
                    mods[k][1][1].params.values,
                    list(mods[k][2][1].params.index),
                    mods[k][2][1].params.values)
            np.save('modelos/'+k+version_model+'.npy', aux)
            print(k+' saved')
        except:
            print(k,'error: model not found')


# from statsmodels.stats.stattools import durbin_watson
# for m in np.arange(38, 53):
#     try:
#         print(m)
#         modp = np.load('modelos/mod'+str(m)+'_objeto.npy', allow_pickle=True)
#         check_normal(modp[0][1].resid)
#         print('D-W: ',durbin_watson(modp[0][1].resid))
#         print()
#     except:
#         print('model not found')
#         print()







