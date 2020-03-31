from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'It works!'

@app.route("/gate", methods=['POST'])
def gate():
    return jsonify({'tags': []}), 200