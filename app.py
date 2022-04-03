import numpy as np
from flask import Flask, request,render_template,jsonify
import pickle


model = pickle.load(open('model_fertilizer.pkl', 'rb'))
app = Flask(__name__)


@app.route('/',methods=['GET'])
def predict():
    
    nitrogen = int(request.args['nitrogen'])
    phosphorus = int(request.args['phosphorus'])
    potassium = int(request.args['potassium'])
    temprature = int(request.args['temprature'])
    humidity = int(request.args['humidity'])
    ph = int(request.args['ph'])
    rainfall = int(request.args['rainfall'])
    pred = model.predict(np.array([nitrogen,phosphorus,potassium,temprature,humidity,ph,rainfall]).reshape(1,-1))
    return jsonify(prediction = str(pred))



if __name__ == "_main_":
    app.run(debug=True)
