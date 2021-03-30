import boto3
import os



def lambda_handler(event, context):
    """
    Estas funciones estan diseñadas para que no pida verificar el usuario ya fue confirmado.
    -------
    event : dict
        Contiene el formato de respuesta de api
    """

    event['response']['autoConfirmUser'] = True
    event['response']['autoVerifyEmail'] = True

    return event

def change_user_group(event,context):
    """
    Estas funciones estan diseñadas para mover a los usuarios de los grupos de cognito.
    -------
    event : dict
        Contiene el formato de respuesta de api
    """

    client = boto3.client('cognito-idp')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ['TABLE_NAME'])

    #Verificamos el grupo que deseamos mover al usuario
    if str(event['request']['userAttributes']['custom:usrl']) == '0':
        group = os.environ['ZERO']
    elif str(event['request']['userAttributes']['custom:usrl']) == '1':
        group = os.environ['ONE']

    #Agregamos el usuario a una tabla en dynamoDB para manejar los usuarios. Si es que se 
    #llega a definir un schema podria cambiarse dynamoDB por RDS
    table.put_item(Item={
            'user_id': event['request']['userAttributes']['custom:company']+"_"+event['request']['userAttributes']['sub'],
            'name': event['request']['userAttributes']['name'],
            'email': event['request']['userAttributes']['email'],
            'position': group,
            })

    #Se cambiar al usuario al nuevo grupo. Estos grupos son definidos en las variables de entorno
    response = client.admin_add_user_to_group(
        UserPoolId = event['userPoolId'],
        Username = event['userName'],
        GroupName= group
        )

    return event
