from flask import Flask,request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Crop = request.form['crop']
        Crop_Year = request.form['crop_year']
        Season = request.form['season']
        State = request.form['state']
        Area = request.form['area']
        Production = request.form['production']
        Annual_Rainfall  = request.form['annual_rainfall']
        Fertilizer  = request.form['fertilizer']
        Pesticide  = request.form['pesticide']

        features = pd.DataFrame([[Crop,Crop_Year, Season, State, Area,Production,Annual_Rainfall,Fertilizer,Pesticide]],
                        columns=['Crop','Crop_Year', 'Season', 'State', 'Area','Production','Annual_Rainfall','Fertilizer','Pesticide'])


    transformed_features = preprocessor.transform(features)
    prediction = dtr.predict(transformed_features).reshape(1,-1)

    return render_template('index.html',prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)