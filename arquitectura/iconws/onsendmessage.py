import json
import os
import sys
import logging
import struct
from decimal import Decimal
sys.path.insert(0, F"{os.environ['LAMBDA_TASK_ROOT']}/{os.environ['DIR_NAME']}")
import boto3
import botocore
from boto3.dynamodb.conditions import Key
sys.path.append("/opt")
import numpy as np
import simplejson
from io import BytesIO

dynamodb = boto3.resource('dynamodb')
lf = boto3.client('lambda')

def _get_response(status_code, body):
    return { 'statusCode': status_code, 'body': body }

def lambda_handler(event, context):
    """
    Funcion que maneja el envio de un mensaje 1 a 1. 
    """

    connections = dynamodb.Table(os.environ['TABLE_NAME'])

    """Variables"""
    data = json.loads(json.loads(event['body'])['data'])

    """Data to make the connection"""

    domain_name = event.get('requestContext',{}).get('domainName')
    actual_user_id = event.get('requestContext',{}).get('connectionId')
    stage = event.get('requestContext',{}).get('stage')

    if (domain_name and stage) is None:
        return _get_response(400, 'Bad Request')


    """Mati Code"""

    ## 100 (tph) 3400 - 3922
    ## 102: medios > 9
    ## 101: medios < 5 y tph < 3600
    ## 103: 5 < medios < 9 y tph > 3600
    ## 100: medios < 5 y tph > 3600
    ## 104 *caso especial --> no va modelo
    if data['tph']>=3400 and data['tph']<3922:
        
        if data['medios']>=9:
            model = 'mod102'
            
        if data['medios']<9:
            if data['tph']>3600 and data['medios']>5:
                model = 'mod103'
                
            if data['tph']>3600 and data['medios']<=5:
                model = 'mod100'
                
            if data['tph']<=3600:
                model = 'mod101'

    ## 201: medios > 8
    ## 200 : medios < 8
    if data['tph']>=3922 and data['tph']<4329:
    
        if data['medios']>8:
            model = 'mod201'

        if data['medios']<=8:
            model = 'mod200'

    ## 301: 8 < velocidad < 9 & porcSolido > 76
    ## 302: 8 < velocidad < 9 & porcSOlido < 76
    ## 300: velocidad > 9 & vel < 8
    if data['tph']>=4329 and data['tph']<4744:
        if 8 < data['velocidad']:
                model = 'mod300'

        if 8 > data['velocidad']:
                model = 'mod301'
    
    ## 400: finos > 80
    ## 401: finos < 80 * finos > 54
    if data['tph']>=4744 and data['tph']<5133:
        model = 'mod4'

    ## 500: medios > 10
    ## 501: medios < 10
    if data['tph']>=5133 and data['tph']<5640:
        model = 'mod5'

    if data['tph']>=5640 and data['tph']<6539:
        model = 'mod6'

    if data['tph']>=6539 and data['tph']<7239:
        model = 'mod7'

    if data['tph']>=7239 and data['tph']<7825:
        model = 'mod8'

    if data['tph']>=7825 and data['tph']<9126:
        model = 'mod9'


    s3 = boto3.resource("s3")
    with BytesIO() as weights:
        s3.Bucket("icon-backend-sag-data").download_fileobj(model, weights)
        weights.seek(0)
        model = np.load(weights, allow_pickle=True)

    index_po, values_po, index_pre, values_pre = model

    potencia = values_po[0]
    presion = values_pre[0]

    #data = {'velocidad': 2, 'porcSolido':3, 'finos':4, 'gruesos':5, 'medios':6,'agua':7, 'tph':8}

    for i_po,v_po in zip(index_po[1:], values_po[1:]):
        temp_i = i_po.split("_")
        potencia += v_po*data[temp_i[0]]**int(temp_i[-1]) if len(temp_i)>1 else v_po*data[temp_i[0]]

    for i_pre, v_pre in zip(index_pre[1:], values_pre[1:]):
        temp_p = i_pre.split("_")
        presion += v_pre*data[temp_p[0]]**int(temp_p[-1]) if len(temp_p)>1 else v_pre*data[temp_p[0]] 

    print(potencia, presion)
    """Response to the front"""

    apigw_management = boto3.client('apigatewaymanagementapi', endpoint_url=F"https://{domain_name}/{stage}")

    try:
        apigw_management.post_to_connection(ConnectionId = actual_user_id, Data = simplejson.dumps({ "potencia":Decimal(potencia), "presion": Decimal(presion)}))
        return _get_response(200, 'Ok')
    except botocore.exceptions.ClientError as e:
            return _get_response(500, 'Something Went Wrong')
