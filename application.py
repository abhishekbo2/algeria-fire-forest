import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application

# import ridge and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Temparature = float(request.form.get('Temparature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data_scale = standard_scaler.transform([[Temparature, RH, WS, Rain, FFMC, DMC, ISI, Classes,region]])

        result = ridge_model.predict(standard_scaler.transform(new_data_scale))

        return render_template("home.html",result=result[0])
    else:
        return render_template('home.html')
    

    return render_template('index.html')
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
