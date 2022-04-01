import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_fertilizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features=np.reshape(final_features, (-1, 1)).T
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='Predicted Crop is {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)
