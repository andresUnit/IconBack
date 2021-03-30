import pandas as pd

def pp_sag(sag1, sag2):
    
    """Función que estructura la data "datos/Base Datos Molino SAG 09092020.xlsx"
    
    Parameters
    -------------
    
    sag1: data de la hoja periodo 1 del archivo "datos/Base Datos Molino SAG 09092020.xlsx"
    
    sag2: data de la hoja periodo 2 del archivo "datos/Base Datos Molino SAG 09092020.xlsx"

    Returns: 
        
        data sag estrucuturada
    -------
    """

    # =========================================================================
    # preprocessing
    # =========================================================================
    # sag1
    sag1 = sag1.iloc[4:, 12:]
    sag1.reset_index(inplace=True, drop=True)
    
    # nombres variables
    names_columns = {
        'velocidad': 'Velocidad Molino SAG - RPM',
        'potencia': 'Potencia Molino - Kwh',
        'presion': 'Presion Promedio Molino SAG - kPa',
        'tph': 'Pesometro Alimentación SAG - TPH',
        'finos': 'Finos correa CV-023 - %',
        'gruesos': 'Gruesos correa CV-023 - %',
        'medios': 'Medios correa CV-023 - %',
        'pebbles': 'Descarga PEBBLES SAG - TPH',
        'impacto_critico': 'Razon de Impacto Critico',
        'nivelpromedio_sp': 'NIVEL STOCKPILE PROMEDIO - %',
        'e_especifica': 'Consumo Especifico Energia SAG - kwh/t',
        'agua': 'ADICION AGUA A SAG MILL m3/h',
        'pocSolido': '% Solido Molino SAG',
        'presion_optima': 'Presion optima SAG5 - kPa',
        'mpc_sag': 'Estatus ON/OFF MPC SAG5'
        }
    
    sag1 = sag1.iloc[:, 0:16]
    sag1 = sag1.iloc[1:]
    sag1.columns = ['fecha', 'velocidad', 'potencia', 'presion', 'tph', 'finos',
                   'gruesos', 'medios', 'pebbles', 'impacto_critico',
                   'nivelpromedio_sp', 'e_especifica', 'agua', 'porcSolido',
                   'presion_optima', 'mpc_sag']
    
    sag1['fecha'] = sag1['fecha'].astype('datetime64[ns]')
    sag1['fecha'].min()
    sag1['fecha'].max()
    
    sag1.set_index('fecha', inplace=True)
    
    # sag2
    sag2 = sag2.iloc[4:, 12:]
    sag2.reset_index(inplace=True, drop=True)
    
    # nombres variables
    sag2 = sag2.iloc[:, 0:16]
    sag2 = sag2.iloc[1:]
    sag2.columns = ['fecha', 'velocidad', 'potencia', 'presion', 'tph', 'finos',
                    'gruesos', 'medios', 'pebbles', 'impacto_critico',
                    'nivelpromedio_sp', 'e_especifica', 'agua', 'porcSolido',
                    'presion_optima', 'mpc_sag']
    
    sag2['fecha'] = sag2['fecha'].astype('datetime64[ns]')
    sag2['fecha'].min()
    sag2['fecha'].max()
    
    sag2.set_index('fecha', inplace=True)
    
    sag = pd.concat([sag1, sag2])
    del sag1
    del sag2
    
    sag.reset_index(inplace=True)
    sag.sort_values('fecha', inplace=True)
    sag.set_index('fecha', inplace=True)
    # temporalidad 9-2019 a 8-2020
    
    return sag

