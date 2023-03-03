import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

WD = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
model = pickle.load(open(os.path.dirname(WD) + '/models/pickled/model.pkl','rb'))


@app.route('/predict-salary', methods=['POST', 'GET'])
def predict():
    # data = request.get_json(force=True)
    data = request.args
    prediction = model.predict([[np.array(eval(data['exp']))]])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
