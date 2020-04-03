import flask
from flask import Flask, jsonify
from gate.predict import predict as predict_gate

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'It works!'

@app.route("/gate", methods=['POST'])
def gate():
    if not flask.request.is_json:
        return jsonify({"error": "JSON payload expected"}), 400

    request = flask.request.get_json()
    if not request["base64"]:
        return jsonify({"error": "base64 image expected"}), 400

    tags = predict_gate(request["base64"])
    return jsonify({'success': True, 'tags': tags}), 200