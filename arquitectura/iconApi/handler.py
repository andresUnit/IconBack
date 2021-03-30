import os
import boto3
import simplejson
from decimal import Decimal
from boto3.dynamodb.conditions import Key
import simplejson

dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    """
    Esta Lambda esta adaptada a la forma de almacenamiento de los pesos en dynamoDB, sugiero cambiar 
    la forma en la que se guardan los pesos a algo mas optimo. Eso por consecuencia haria cambiar esta
    funcion.
    -------
    response : dict
        Contiene el formato de respuesta de api
    """

    table = dynamodb.Table(os.environ['TABLE_NAME'])
    t_response = {}

    models = table.get_item(Key={'model':'Modelos'})['Item']['value']
    for model in models:
        coef = []
        return_data = table.get_item(Key={'model':model})['Item']['value']
        data_keys = sorted(return_data.keys())
        temporal_dict = {}
        for dk in data_keys:
            if dk.split("_")[0] not in temporal_dict:
                temporal_dict[dk.split("_")[0]] = []
            temporal_dict[dk.split("_")[0]].append(Decimal(return_data[dk]))
        for dk in temporal_dict:
            coef.append({'name': dk, "value": temporal_dict[dk]})
        t_response[model] = coef

    response = {"statusCode": str(200), 
                    "body": simplejson.dumps(t_response),                 
                    "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": '*'
                            }
                    }
    return response
