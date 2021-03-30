import pandas as pd
import numpy as np
import datetime as dt
import warnings as wa

# from estr_sag import pp_sag
# from g_estacionario import g_estacionario
# from g_tph import g_tph
# from g_weekSpiTphNSP import (g_nps,
#                              sag_estacionario_week,
#                              g_tph_week,
#                              valores_optimos)
# from m_polinomial import (sim_gtph,
#                           sim_gspi)

# wa.filterwarnings('ignore')

'''
    La data sag esta albergado en el archivo .xlsx en 
    "datos/Base Datos Molino SAG 09092020.xlsx" donde contiene la data en
    2 hojas llamadas 'Periodo 1' y 'Periodo 2'.
'''

def orq():
    
    '''
    Orquestador: Controla la arquitectura y flujo de códigos:
        
        - Estructuración de la data.
        - Preprocessing.
        - Modelamiento.
        
    * Debe ser controlado en Spyder sin usarlo como función.
'''

    # =========================================================================
    # Estructuración
    # =========================================================================
    # periodo 1
    sag1 = pd.read_excel('../../../datos/Base Datos Molino SAG 09092020.xlsx',
                         sheet_name='Periodo 1')
    # periodo 2
    sag2 = pd.read_excel('../../../datos/Base Datos Molino SAG 09092020.xlsx',
                         sheet_name='Periodo 2')
    
    sag = pp_sag(sag1, sag2)
    sag.to_pickle('../../../datos/sag22.pkl')
    print('structured data saved /../../../datos/sag.pkl')
    del sag
    
    # =========================================================================
    # Preprocessing
    # =========================================================================
    sag = pd.read_pickle('../../../../datos/sag.pkl')
    sag_estacionario = g_estacionario(sag)
    sag_estacionario.to_pickle('../../../../datos/sag_estacionario1.pkl')
    
    print('data sag saved /../../../datos/sag_estacionario1.pkl')
    del sag_estacionario
    
    # g_tph
    sag_estacionario = pd.read_pickle('../../../../datos/sag_estacionario1.pkl')
    sag_estacionario = g_tph(sag_estacionario)
    sag_estacionario.to_pickle('../../../../datos/sag_estacionario2.pkl')
    print('data sag saved /../../../../datos/sag_estacionario2.pkl')
    del sag_estacionario
    
    # g_spi y optimos
    sag_estacionario = pd.read_pickle('../../../../datos/sag_estacionario2.pkl')
    
    sag_estacionario = g_nps(sag_estacionario)
    sag_estacionario_w = sag_estacionario_week(sag_estacionario)
    sag_estacionario, info_grupos = g_tph_week(sag_estacionario, sag_estacionario_w)
    sag_estacionario, inicial_optimo, info_gspi_tph = valores_optimos(sag_estacionario,
                                                                   info_grupos['kmeans'])
    sag_estacionario.to_pickle('../../../../datos/sag_estacionario3.pkl')
    info_gspi_tph.to_excel('reportes/info_gspi_tph.xlsx')
    np.save('reportes/info_grupos_tphweek.npy', info_grupos)
    np.save('reportes(inicial_optimo.npy', inicial_optimo)
    
    print('agregado categorias spi exportado a "../../../../datos/sag_estacionario3.pkl"')
    print('grupos spi con grupos tph (info_gspi_tph) exportado a "reportes/info_gspi_tph.xlsx"')
    print('estadisticas grupos tph por semana exportado a "reportes/info_grupos_tphweek.npy"')
    print('valores optimos e iniciales aleatorios para el front exportado en "parametros/inicial_optimo.npy"')
    
    # =========================================================================
    # Modelamiento
    # =========================================================================
    # ** HAY QUE VER SI COINCIDEN LOS MODELOS ** 
    sag_estacionario = pd.read_pickle('../../../../datos/sag_estacionario3.pkl')
    
    sim_gtph(sag_estacionario, train_date_until=dt.datetime(2020, 7, 1),
             version_model=1)
    sim_gspi(sag_estacionario, train_date_until=dt.datetime(2020, 2, 14),
             version_model=1)


