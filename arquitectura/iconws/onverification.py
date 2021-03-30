import json
import os
import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
client = boto3.client('cognito-idp')
lf = boto3.client('lambda')

def _get_response(status_code, body):
    if not isinstance(body, str):
        body = json.dumps(body)
    return {"statusCode": status_code, "body": body}

def lambda_handler(event, context):
    """
    Esta funcion verifica que el usuario que a hecho el handshake sea un usuario registrado en cognito utilizando la 
    lambda function authorizer. Esto se hace por que no podemos agregar headers utilizando ws. 
    -------
    _get_response : dict
        Contiene el formato de respuesta de api
    """

    connectionID = event["requestContext"].get("connectionId")
    jwtToken = json.loads(event.get('body', '{}')).get('jwtToken')
    response = lf.invoke(
        FunctionName = os.environ['HANDLER_ARN'],
        InvocationType = "RequestResponse",
        Payload = json.dumps({"token": jwtToken}))
    response = json.loads(response['Payload'].read().decode("utf-8"))
    if response['statusCode'] == 200 and 'custom:company' in response['body']:
        try: 
            table = dynamodb.Table(os.environ['TABLE_NAME'])
            #Actualizamos en dynamo agregando datos que provienente del id token. Esto indica al websocket
            #que es un usuario de cognito.
            table.update_item(Key={'tableId':connectionID},
                    UpdateExpression="set tableValue=:u, companyName=:c",
                    ExpressionAttributeValues={
                        ':u': response['body']['cognito:username'],
                        ':c': response['body']['custom:company']
                    },
                    ReturnValues="UPDATED_NEW"
                )
            table.put_item(Item={'tableId': response['body']['cognito:username'],'tableValue': connectionID, 'companyName': response['body']['custom:company']})
            return _get_response(200, "Verification successful.")
        except:
            return _get_response(400, "Bad request")
    else:
        return _get_response(400, "Bad request")