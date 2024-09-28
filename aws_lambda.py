import json
import joblib
import numpy as np

# Load the model when the Lambda function is initialized
model = joblib.load("model.pkl")

def lambda_handler(event, context):
    # Extract features from the incoming event (assuming JSON format)
    features = event['features']
    features_array = np.array(features).reshape(1, -1)

    # Perform the prediction
    prediction = model.predict(features_array)

    # Return the prediction
    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': prediction
        })
    }


