import yaml
import sys
import ast
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from PIL import Image
from flask import Flask
import requests
import torch
import json
import wget
import aiohttp
import asyncio
import uvicorn
import nest_asyncio
import warnings
import gdown
from torchvision import transforms
warnings.filterwarnings("ignore")
print("Imports Done")

nest_asyncio.apply()
export_file_url = 'https://drive.google.com/u/0/uc?id=1x5Ljh9xtNfXFMm97AMlewW1nZ77XS-gb&export=download'
export_file_name = 'cattle_breed_classifier_full_model.pth'
path = Path("models/")
export_classes_url = "https://drive.google.com/u/0/uc?id=1IaF_zn-RDnsEntYp86F5G7FNlEkQ8KJ_&export=download"
export_classes_name = "classes.txt"

async def download_file(url, dest):
    if dest.exists(): return
    gdown.download(url, str(dest), quiet=False)
                
async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    await download_file(export_classes_url, path / export_classes_name)
    try:
        learn = torch.load(path / export_file_name, map_location=torch.device('cpu'))
        with open(path / export_classes_name, 'r') as file:
             class_list = file.read().split(",")
        return learn, class_list
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "Something went wrong!"
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
model, class_list = loop.run_until_complete(asyncio.gather(*tasks))[0]

with open("config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)


def load_model(path=".", model_name="cattle_breed_classifier_full_model.pth"):
    learn = torch.load(export_file_name, map_location=torch.device('cpu'))
    return learn


def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = Image.open(BytesIO(raw_bytes))
    return img


def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    outputs = model(transform(img).unsqueeze(0)).squeeze()
    pred_probs = torch.nn.Softmax(dim=-1)(outputs)
    _, pred_class = torch.max(pred_probs, dim=0)
    pred_probs = pred_probs.tolist()
    predictions = []
    for image_class, output, prob in zip(class_list, outputs.tolist(), pred_probs):
        output = round(output, 1)
        prob = round(prob, 2)
        predictions.append(
            {"class": image_class.replace("_", " "), "output": output, "prob": round(prob, 2)}
        )
    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
    predictions = predictions[0:n]
    print({"class": str(pred_class.item()), "predictions": predictions})
    return {"class": str(pred_class.item()), "predictions": predictions}


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
    else:
        bytes = flask.request.files['file'].read()
        img = load_image_bytes(bytes)
    res = predict(img)
    return flask.jsonify(res)


@app.route('/api/classes', methods=['GET'])
def classes():
    with open('models/classes.txt', 'r') as file:
        classes = file.read().split(",")
    return flask.jsonify(classes)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}


if __name__ == '__main__':
    port = os.environ.get('PORT', 5001)

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=False, host='0.0.0.0', port=port)
        # app.run(host='0.0.0.0', port=port)
