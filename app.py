import numpy as np
from flask import Flask, request,render_template,jsonify
import pickle


model = pickle.load(open('model_fertilizer.pkl', 'rb'))
app = Flask(__name__)



@app.route('/',methods=['GET'])
def predict():
    
    nitrogen = float(request.args['nitrogen'])
    phosphorus = float(request.args['phosphorus'])
    potassium = float(request.args['potassium'])
    temprature = float(request.args['temprature'])
    humidity = float(request.args['humidity'])
    ph = float(request.args['ph'])
    rainfall = float(request.args['rainfall'])
    pred = model.predict(np.array([nitrogen,phosphorus,potassium,temprature,humidity,ph,rainfall]).reshape(1,-1))
    return jsonify(prediction = str(pred))



if __name__ == "_main_":
    app.run(debug=True)
