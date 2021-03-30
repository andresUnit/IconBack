
# model = load_model("modelos/descarga_torque.h5")

# recibe estado y retorna la actualizacion del estado
def simulacion(state, model, change = False ):
    """Función que testea la simulacion.
    
    state: lista escalada de estado incial del espesador.
    model: modelo de redes lstm.
    change: 

    Parameters
    -------------
    
    db_epesadores: data del archivo "datos/Base Datos Espesadores de Relaves 09092020.xlsx"
    
    Returns: 
        
        Tupla con los datos train_x, train_y, test_x, test_y.
    -------
    """
    if change:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        elemento = input("Ingrese la Variable (ap,dp,fl,ps): ")
        cantidad = input("Ingrese Cantidad: ")
        elementos = ["ap", "dp", "fl", "ps"]
        
        index = elementos.index(elemento)+1
        state_ini_rescaled[0][index]=float(cantidad)
        state_aux = scaler.transform(state_ini_rescaled)
        state_aux = state_aux.flatten()
        pred = model.predict(state_aux.reshape((1,1,-1)))
        print(pred.shape)
        variacion = (pred-state_aux[0])*100/state_aux[0]
        state_aux[0]=pred
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state_aux.reshape(1,-1))
        print_data(state_fin_rescaled)
        density = 1/(state_fin_rescaled[0][4]/2.75-(1-state_fin_rescaled[0][4]))
        flujoM = state_fin_rescaled[0][1]*density*state_fin_rescaled[0][4]
        print("Flujo Masico: ")
        print(flujoM)
        print("Variacion de Torque: ")
        print(variacion[0][0])
        return state_aux
        
    else:
        print("Estado Inicial")
        state_ini_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_ini_rescaled)
        pred = model.predict(state.reshape((1,1,-1)))
        variacion = (pred-state[0])*100/state[0]
        state[0]=pred
        print("Estado Final")
        state_fin_rescaled = scaler.inverse_transform(state.reshape(1,-1))
        print_data(state_fin_rescaled)
        density = 1/(state_fin_rescaled[0][4]/2.75-(1-state_fin_rescaled[0][4]))
        flujoM = state_fin_rescaled[0][1]*density*state_fin_rescaled[0][4]
        print("Flujo Masico: ")
        print(flujoM)
        print("Variacion de Descarga de Torque: ")
        print(variacion[0][0])
    return state

