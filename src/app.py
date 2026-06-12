import os
import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def _resolve_model_path():
    bucket = os.environ.get('MODEL_BUCKET')
    if not bucket:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
    local_path = '/tmp/model.pt'
    if not os.path.exists(local_path):
        import boto3
        key = os.environ.get('MODEL_KEY', 'model.pt')
        boto3.client('s3').download_file(bucket, key, local_path)
        print(f"Model downloaded from s3://{bucket}/{key}")
    return local_path

model = CNNClassifier()
try:
    model.load_state_dict(torch.load(_resolve_model_path(), map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()


def run_inference(pixels_list):
    if not isinstance(pixels_list, list) or len(pixels_list) != 784:
        return None, 'Invalid input: expecting a list of 784 pixel values.'
    try:
        pixels = np.array(pixels_list, dtype=np.float32).reshape(1, 28, 28)
    except ValueError:
        return None, 'Pixel data could not be reshaped to 28x28.'

    pixels = (pixels - 0.5) / 0.5
    input_tensor = torch.from_numpy(pixels).unsqueeze(0)

    activations = {}
    def make_hook(name):
        def hook(module, inp, output):
            activations[name] = output.detach()
        return hook

    h1 = model.layers[3].register_forward_hook(make_hook('conv1'))
    h2 = model.layers[7].register_forward_hook(make_hook('conv2'))

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.item()

    h1.remove()
    h2.remove()

    confidences = torch.softmax(outputs, dim=1).squeeze().tolist()

    def process_maps(act):
        act = act.squeeze(0).numpy()
        result = []
        for i in range(act.shape[0]):
            fm = act[i]
            lo, hi = float(fm.min()), float(fm.max())
            fm = (fm - lo) / (hi - lo) if hi > lo else np.zeros_like(fm)
            result.append(fm.tolist())
        return result

    return {
        'prediction': prediction,
        'confidences': confidences,
        'conv1': process_maps(activations['conv1']),
        'conv2': process_maps(activations['conv2']),
    }, None


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result, error = run_inference(data.get('pixels') if data else None)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)


def lambda_handler(event, context):
    cors_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': cors_headers, 'body': ''}

    try:
        body = json.loads(event.get('body') or '{}')
        result, error = run_inference(body.get('pixels'))
        if error:
            return {'statusCode': 400, 'headers': cors_headers, 'body': json.dumps({'error': error})}
        return {'statusCode': 200, 'headers': cors_headers, 'body': json.dumps(result)}
    except Exception as e:
        return {'statusCode': 500, 'headers': cors_headers, 'body': json.dumps({'error': str(e)})}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
