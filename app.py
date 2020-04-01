from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'It works!'

@app.route("/gate", methods=['POST'])
def gate():
    return jsonify({'success': True, 'tags': [{'tag': 'closed', 'confidence': 0.9}, {'tag': 'open', 'confidence': 0.1}]}), 200