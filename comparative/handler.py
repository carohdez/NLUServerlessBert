try:
    import unzip_requirements
except ImportError:
    pass
from model.model import ServerlessModel
import json

model = ServerlessModel('./model', 'clf-model', 'conversational/intention/scope/comparative.tar.gz')

def predict_comparative(event, context):
    try:
        body = json.loads(event['body'])
        print("I'm about to predict comparative")
        type = model.predict(body['sentence'])
        return {
        "statusCode": 200,
        "headers": {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        "Access-Control-Allow-Credentials": True

        },
        "body": json.dumps({'type': type})
        }
    except Exception as e:
        return {
        "statusCode": 500,
        "headers": {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        "Access-Control-Allow-Credentials": True
        },
        "body": json.dumps({"error": repr(e)})
        }