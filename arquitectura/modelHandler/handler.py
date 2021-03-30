import json
import sys

sys.path.append("/opt")
import pandas as pd

def lambda_handler(event, context):
    # TODO implement
    df = pd.DataFrame()
    print(pd.__version__)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }